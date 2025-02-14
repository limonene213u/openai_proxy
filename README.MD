# OpenAI API プロキシサーバー

このプロジェクトは、OpenAI APIのプロキシサーバーとして機能し、トークン使用量の追跡と記録を行います。

## 機能

- OpenAI APIへのリクエストのプロキシ
- トークン使用量の追跡とログ記録（InfluxDBとSQLite）
- ユーザーごとのAPIレートリミット
- Prometheusメトリクスのエクスポート
- ユーザーごとの使用履歴の取得

## 必要条件

- Go 1.16以上
- SQLite3
- InfluxDB（オプション、使用量のログ記録用）

## セットアップ

1. リポジトリをクローンします：

   ```
   git clone https://github.com/yourusername/openai-api-proxy.git
   cd openai-api-proxy
   ```

2. 依存関係をインストールします：

   ```
   go mod tidy
   ```

3. `.env`ファイルを作成し、必要な環境変数を設定します：

   ```
   cp .env.example .env
   ```

   `.env`ファイルを編集し、必要な情報を入力します。

4. プログラムをビルドします：

   ```
   go build -o proxy-server
   ```

## 使用方法

1. プロキシサーバーを起動します：

   ```
   ./proxy-server
   ```

2. クライアントからプロキシサーバーにリクエストを送信します。例：

   ```
   curl -X POST "http://localhost:8080/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -H "X-User-ID: user123" \
        -d '{
          "model": "gpt-3.5-turbo",
          "messages": [{"role": "user", "content": "Hello!"}]
        }'
   ```

3. 使用履歴を確認するには：

   ```
   curl "http://localhost:8080/usage?user_id=user123"
   ```

4. Prometheusメトリクスを確認するには：

   ```
   curl "http://localhost:8080/metrics"
   ```

## 設定

環境変数を使用して設定を行います。`.env`ファイルまたはシステムの環境変数で設定できます：

- `OPENAI_API_KEY`: OpenAI APIキー
- `PROXY_TARGET`: プロキシ先のURL（通常は`https://api.openai.com`）
- `INFLUXDB_URL`: InfluxDBのURL
- `MODEL_NAME`: デフォルトで使用するAIモデル名
- `PORT`: サーバーのポート番号（デフォルトは8080）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。