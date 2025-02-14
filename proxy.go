package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
)

var (
	proxyAPIKey      string
	influxURL        string
	proxyTarget      string
	modelName        string
	totalTokenMetric int64
	db               *sql.DB
	httpClient       *http.Client
	rateLimiter      = make(map[string]*RateLimiter)
	rateLimiterMutex sync.Mutex
)

type RateLimiter struct {
	count      int
	lastUpdate time.Time
}

func init() {
	if err := godotenv.Load(); err != nil {
		log.Printf("警告: .env ファイルの読み込みに失敗しました: %v", err)
	}

	proxyAPIKey = os.Getenv("OPENAI_API_KEY")
	influxURL = os.Getenv("INFLUXDB_URL")
	proxyTarget = os.Getenv("PROXY_TARGET")
	modelName = os.Getenv("MODEL_NAME")

	if proxyTarget == "" {
		log.Fatal("PROXY_TARGET 環境変数が設定されていません")
	}

	var err error
	db, err = sql.Open("sqlite3", "./tokens.db")
	if err != nil {
		log.Fatalf("SQLiteデータベース接続エラー: %v", err)
	}

	httpClient = &http.Client{
		Timeout: time.Second * 10,
	}

	if err := createTables(); err != nil {
		log.Fatalf("テーブル作成エラー: %v", err)
	}
}

func createTables() error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS usage (
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			user_id TEXT,
			model TEXT,
			prompt_tokens INT,
			completion_tokens INT,
			total_tokens INT
		);
		CREATE TABLE IF NOT EXISTS cumulative_usage (
			user_id TEXT PRIMARY KEY,
			total_tokens INT DEFAULT 0
		);
	`)
	return err
}

func main() {
	target, err := url.Parse(proxyTarget)
	if err != nil {
		log.Fatalf("プロキシターゲットのURL解析エラー: %v", err)
	}
	proxy := httputil.NewSingleHostReverseProxy(target)

	http.HandleFunc("/", handleProxy(proxy))
	http.HandleFunc("/metrics", handleMetrics)
	http.HandleFunc("/usage", handleUsage)

	server := &http.Server{
		Addr:    ":8080",
		Handler: nil,
	}

	go func() {
		log.Println("サーバーを起動しています...")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("サーバーエラー: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("シャットダウンを開始します...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("サーバーのグレースフルシャットダウンに失敗しました: %v", err)
	}

	log.Println("サーバーを停止しました")
}

func handleProxy(proxy *httputil.ReverseProxy) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := r.Header.Get("X-User-ID")
		if userID == "" {
			userID = "unknown"
		}

		if !checkRateLimit(userID) {
			http.Error(w, "レートリミット超過", http.StatusTooManyRequests)
			return
		}

		clientAPIKey := r.Header.Get("Authorization")
		if clientAPIKey == "" && proxyAPIKey != "" {
			r.Header.Set("Authorization", "Bearer "+proxyAPIKey)
		}

		responseRecorder := &ResponseRecorder{
			ResponseWriter: w,
			userID:         userID,
		}

		startTime := time.Now()
		proxy.ServeHTTP(responseRecorder, r)
		elapsedTime := time.Since(startTime)

		log.Printf("APIレスポンス時間: %s", elapsedTime)
		log.Printf("レスポンスステータス: %d", responseRecorder.statusCode)
	}
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	currentTotal := atomic.LoadInt64(&totalTokenMetric)
	fmt.Fprintf(w, "proxy_total_tokens %d\n", currentTotal)
}

func handleUsage(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "ユーザーIDが必要です", http.StatusBadRequest)
		return
	}

	usage, err := getRecentUsage(r.Context(), userID)
	if err != nil {
		log.Printf("使用履歴の取得に失敗しました: %v", err)
		http.Error(w, "使用履歴の取得に失敗しました", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(usage); err != nil {
		log.Printf("JSON エンコードエラー: %v", err)
		http.Error(w, "内部サーバーエラー", http.StatusInternalServerError)
	}
}

type ResponseRecorder struct {
	http.ResponseWriter
	statusCode int
	body       []byte
	userID     string
}

func (r *ResponseRecorder) WriteHeader(statusCode int) {
	if r.statusCode == 0 {
		r.statusCode = statusCode
	}
	r.ResponseWriter.WriteHeader(statusCode)
}

func (r *ResponseRecorder) Write(body []byte) (int, error) {
	r.body = append(r.body, body...)

	promptTokens, completionTokens, totalTokens := extractTokenUsage(r.body)
	log.Printf("トークン使用量: Prompt=%d, Completion=%d, Total=%d", promptTokens, completionTokens, totalTokens)

	atomic.AddInt64(&totalTokenMetric, int64(totalTokens))

	logToAsyncAll(r.userID, promptTokens, completionTokens, totalTokens)

	return r.ResponseWriter.Write(body)
}

func extractTokenUsage(body []byte) (int, int, int) {
	var data struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	err := json.Unmarshal(body, &data)
	if err != nil {
		log.Printf("JSON解析エラー: %v, レスポンスの一部: %s", err, string(body[:min(len(body), 200)]))
		return 0, 0, 0
	}
	return data.Usage.PromptTokens, data.Usage.CompletionTokens, data.Usage.TotalTokens
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func logToAsyncAll(userID string, promptTokens, completionTokens, totalTokens int) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := logToInfluxDB(ctx, promptTokens, completionTokens, totalTokens); err != nil {
			log.Printf("InfluxDBへのログ記録に失敗しました: %v", err)
		}
		if err := logToSQLite(ctx, userID, promptTokens, completionTokens, totalTokens); err != nil {
			log.Printf("SQLiteへのログ記録に失敗しました: %v", err)
		}
		if err := updateCumulativeTokens(ctx, userID, totalTokens); err != nil {
			log.Printf("累積トークン更新に失敗しました: %v", err)
		}
	}()
}

func logToInfluxDB(ctx context.Context, promptTokens, completionTokens, totalTokens int) error {
	if influxURL == "" {
		return fmt.Errorf("InfluxDB URLが設定されていません")
	}
	useModel := modelName
	if useModel == "" {
		useModel = "gpt-4"
	}
	data := fmt.Sprintf("tokens,model=%s prompt_tokens=%d,completion_tokens=%d,total_tokens=%d",
		useModel, promptTokens, completionTokens, totalTokens)
	req, err := http.NewRequestWithContext(ctx, "POST", influxURL+"/write?db=token_usage", strings.NewReader(data))
	if err != nil {
		return fmt.Errorf("InfluxDBリクエスト作成エラー: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("InfluxDBへの送信エラー: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("InfluxDBが予期しないステータスコードを返しました: %d", resp.StatusCode)
	}

	log.Println("トークンデータをInfluxDBに記録しました")
	return nil
}

func logToSQLite(ctx context.Context, userID string, promptTokens, completionTokens, totalTokens int) error {
	useModel := modelName
	if useModel == "" {
		useModel = "gpt-4"
	}

	_, err := db.ExecContext(ctx, `
		INSERT INTO usage (user_id, model, prompt_tokens, completion_tokens, total_tokens)
		VALUES (?, ?, ?, ?, ?)
	`, userID, useModel, promptTokens, completionTokens, totalTokens)
	if err != nil {
		return fmt.Errorf("SQLite挿入エラー: %w", err)
	}

	log.Println("トークンデータをSQLiteに記録しました")
	return nil
}

func updateCumulativeTokens(ctx context.Context, userID string, totalTokens int) error {
	_, err := db.ExecContext(ctx, `
		INSERT INTO cumulative_usage (user_id, total_tokens)
		VALUES (?, ?)
		ON CONFLICT(user_id) DO UPDATE SET total_tokens = total_tokens + ?
	`, userID, totalTokens, totalTokens)
	if err != nil {
		return fmt.Errorf("累積トークン更新エラー: %w", err)
	}
	return nil
}

func getRecentUsage(ctx context.Context, userID string) ([]map[string]interface{}, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT timestamp, prompt_tokens, completion_tokens, total_tokens
		FROM usage
		WHERE user_id = ?
		ORDER BY timestamp DESC
		LIMIT 10
	`, userID)
	if err != nil {
		return nil, fmt.Errorf("使用履歴クエリエラー: %w", err)
	}
	defer rows.Close()

	var usage []map[string]interface{}
	for rows.Next() {
		var timestamp string
		var promptTokens, completionTokens, totalTokens int
		if err := rows.Scan(&timestamp, &promptTokens, &completionTokens, &totalTokens); err != nil {
			return nil, fmt.Errorf("行のスキャンエラー: %w", err)
		}
		usage = append(usage, map[string]interface{}{
			"timestamp":         timestamp,
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      totalTokens,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("行の反復エラー: %w", err)
	}
	return usage, nil
}

func checkRateLimit(userID string) bool {
	rateLimiterMutex.Lock()
	defer rateLimiterMutex.Unlock()

	limiter, exists := rateLimiter[userID]
	if !exists {
		rateLimiter[userID] = &RateLimiter{count: 1, lastUpdate: time.Now()}
		return true
	}

	if time.Since(limiter.lastUpdate) > time.Minute {
		limiter.count = 1
		limiter.lastUpdate = time.Now()
		return true
	}

	if limiter.count >= 10 {
		return false
	}

	limiter.count++
	return true
}
