# Troubleshooting Guide

## Common Issues and Solutions

### 1. Model Loading Errors
**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check Python version compatibility (TensorFlow supports Python 3.8-3.11)
- For Python 3.12+, use TensorFlow 2.16+ or consider using a different Python version

### 2. Image Processing Errors
**Error:** `PIL.UnidentifiedImageError` or `OSError: cannot identify image file`

**Solution:**
- Verify the image file is not corrupted
- Supported formats: JPG, JPEG, PNG
- Check file size (max 10MB)
- Ensure the image has valid RGB channels

### 3. API Authentication Issues
**Error:** `401 Unauthorized`

**Solution:**
- Include `Authorization: Bearer <API_KEY>` header
- Development fallback key is `dev-key-not-for-production` (NEVER use in production!)
- Set `API_KEY` environment variable for production: `export API_KEY=your-secret-key`
- In production mode (`ENVIRONMENT=production`), the server will refuse to start without API_KEY

### 4. ngrok Tunnel Issues

**Error:** `ERR_NGROK_4018: Your ngrok agent is not connected to a valid ngrok account`
— OR — `ERR_NGROK_108: authtoken is not valid`
— OR — ngrok tunnel silently fails / no public URL shown

**Root Cause:** ngrok requires a free authtoken to create tunnels. Without it, `ngrok http 8000` will fail immediately.

**Fix (one-time setup, ~2 minutes):**

1. **Sign up** for a free account at: https://dashboard.ngrok.com/signup
2. **Copy your authtoken** from: https://dashboard.ngrok.com/get-started/your-authtoken
3. **Configure ngrok** — open a terminal and run:
   ```
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```
4. **Re-run** `auto_start.bat` — the tunnel should now work.

**Other common ngrok issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ERR_NGROK_108` | Authtoken expired/revoked | Get a new token from the dashboard and run `ngrok config add-authtoken NEW_TOKEN` |
| `ERR_NGROK_4022` | Too many tunnels | Free plan allows 1 tunnel. Close other tunnels or kill old ngrok: `taskkill /F /IM ngrok.exe` |
| URL changes every restart | Free plan uses random URLs | Upgrade to a paid plan for fixed domains, or just copy the new URL each time |
| `bind: address already in use` | Port 8000 occupied | Kill the process using port 8000: `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| Tunnel works but page shows "Visit blocked" | ngrok interstitial page | Click "Visit Site" on the ngrok warning page. This is normal for free ngrok plans. |
| Tunnel silently fails / no URL shown | Config file uses old `version: '2'` format | Run `ngrok config upgrade` to migrate to v3 format. `auto_start.bat` does this automatically. |

**Verifying ngrok is working:**
```bash
# Check if ngrok tunnel is active:
curl http://localhost:4040/api/tunnels

# Check ngrok config location:
ngrok config check
```

### 5. Rate Limiting
**Error:** `429 Too Many Requests`

**Solution:**
- Wait before retrying (rate limit: 10 requests/minute for single predictions)
- Implement exponential backoff in client code
- Consider upgrading to paid tier for higher limits

### 6. Memory Issues
**Error:** `MemoryError` or out-of-memory crashes

**Solution:**
- Reduce batch size for `/predict/batch` endpoint (max 20 files)
- Ensure sufficient RAM (recommend 4GB+)
- Process images sequentially instead of parallel

### 7. Docker Build Issues
**Error:** Build fails with OpenCV dependencies

**Solution:**
- Ensure Docker has sufficient memory allocated
- Use the provided Dockerfile which includes all required system libs
- For Apple Silicon Macs, add `--platform=linux/amd64` to build command

### 8. Streamlit App Won't Start
**Error:** Port already in use or connection refused

**Solution:**
- Kill existing processes: `pkill -f streamlit`
- Change port: `streamlit run app.py --server.port 8502`
- Check firewall settings

### 9. Model Prediction Inconsistencies
**Issue:** Different results for same image

**Solution:**
- Model predictions can vary slightly due to floating-point precision
- Ensure consistent image preprocessing
- Check if model was retrained (compare model file hash)

## Performance Optimization

### For API Performance:
- Use Gunicorn with multiple workers: `gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker`
- Enable model caching in production
- Use faster image processing libraries if needed

### For Streamlit Performance:
- Run in headless mode for production
- Use `streamlit run app.py --server.headless true`
- Implement session state cleanup

## Getting Help

1. Check the [README.md](README.md) for setup instructions
2. Review [API documentation](http://localhost:8000/docs) when running
3. Check GitHub Issues for similar problems
4. Include full error traceback and system information when reporting bugs

## System Requirements

- **Python:** 3.10+ recommended (3.12 supported with TensorFlow 2.16+)
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 2GB for model and dependencies
- **OS:** Linux/macOS/Windows (Linux recommended for production)