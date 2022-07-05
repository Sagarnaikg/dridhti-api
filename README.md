## Backend API

Our backend is built using `flask` library and 
deployed on `heroku` server

```bash
app = Flask(__name__)

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    ....
```

## API Reference

Getting the API connection status

```http
  GET /
```
Retruns
| Key | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `message` | `string` | API connection status |

Post an Image and get a caption

```http
  PUT /upload
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `image`      | `file` | **Required** Image file  |

Retruns
| Key | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `result` | `string` | Image caption |
