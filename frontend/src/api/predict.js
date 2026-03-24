/**
 * API client for Brand Vision backend.
 * Never hardcodes the API URL — always reads from VITE_API_URL env var.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Send an image file to the /predict endpoint.
 * @param {File} file - The image file to classify
 * @returns {Promise<Object>} Prediction result JSON
 * @throws {Error} If the request fails or the server returns an error
 */
export async function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let detail = 'Prediction failed'
    try {
      const err = await response.json()
      detail = err.detail || detail
    } catch (_) {
      // Ignore JSON parse error on error responses
    }
    throw new Error(detail)
  }

  return response.json()
}
