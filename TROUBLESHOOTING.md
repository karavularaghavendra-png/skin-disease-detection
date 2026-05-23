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

### 4. Rate Limiting
**Error:** `429 Too Many Requests`

**Solution:**
- Wait before retrying (rate limit: 10 requests/minute for single predictions)
- Implement exponential backoff in client code
- Consider upgrading to paid tier for higher limits

### 5. Memory Issues
**Error:** `MemoryError` or out-of-memory crashes

**Solution:**
- Reduce batch size for `/predict/batch` endpoint (max 20 files)
- Ensure sufficient RAM (recommend 4GB+)
- Process images sequentially instead of parallel

### 6. Docker Build Issues
**Error:** Build fails with OpenCV dependencies

**Solution:**
- Ensure Docker has sufficient memory allocated
- Use the provided Dockerfile which includes all required system libs
- For Apple Silicon Macs, add `--platform=linux/amd64` to build command

### 7. Streamlit App Won't Start
**Error:** Port already in use or connection refused

**Solution:**
- Kill existing processes: `pkill -f streamlit`
- Change port: `streamlit run app.py --server.port 8502`
- Check firewall settings

### 8. Model Prediction Inconsistencies
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