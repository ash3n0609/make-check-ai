# Maker-Checker AI

A dual-model AI system that uses a "Maker" model to generate responses and a "Checker" model to critique and improve them. Built with Modal and Firebase.

## Setup Instructions

### 1. Prerequisites
- [Modal](https://modal.com/) account and CLI installed (`pip install modal`).
- [Firebase](https://firebase.google.com/) project for chat persistence.
- API keys for Kimi (NVIDIA), DeepSeek, or Gemini (if using online models).

### 2. Configuration

#### Initial Configuration
The project uses `.example` files for configuration. Before starting, copy them to their actual names which are ignored by git:

```bash
cp config.example.yaml config.yaml
cp config.example.js config.js
```

#### Backend (`config.yaml`)
Edit `config.yaml` to specify your Modal username and desired app name:
```yaml
modal:
  username: "your-modal-username"
  app_name: "maker-checker-demo"
  endpoint_label: "maker-checker"
  get_chats_label: "get-user-chats"
  get_history_label: "get-chat-history"
models:
  maker: "allenai/OLMo-2-1124-7B-Instruct"
  checker: "Qwen/Qwen3-4B"
```

#### Environment Variables (`.env`)
Create a `.env` file in the root directory and add your API keys:
```env
DEEPSEEK_API_KEY="your-key"
KIMI_API_KEY="your-key"
GEMINI_API_KEY="your-key"
```

#### Firebase
- Add your `serviceAccountKey.json` (server-side) to the root directory.
- Update `config.js` with your Firebase web configuration (client-side).

### 3. Deployment

Deploy the backend to Modal:
```bash
modal deploy path.py
```

After deployment, Modal will provide several URLs. Look for the one ending in the `endpoint_label` you specified (default: `maker-checker`).
Example: `https://your-username--maker-checker.modal.run`

### 4. Frontend Configuration

Open `config.js` and paste your deployed Modal endpoint and your Firebase web config:
```javascript
window.CONFIG = {
    endpoint: "https://your-username--maker-checker.modal.run",
    firebaseConfig: {
        apiKey: "...",
        // ... (rest of your firebase config)
    }
};
```

### 5. Running the App
You can now open `index.html` in any browser to use the application.

## Project Structure
- `path.py`: Modal backend definition.
- `index.html`: Frontend interface.
- `config.yaml`: Backend configuration (ignored by git).
- `config.js`: Frontend configuration (ignored by git).
- `config.example.yaml`: Template for backend configuration.
- `config.example.js`: Template for frontend configuration.
- `chat_service.py`: Firebase integration for chat persistence.
