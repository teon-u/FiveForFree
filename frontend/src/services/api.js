import axios from 'axios'

// Create axios instance with default config
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 30000, // Increased timeout for slow API responses
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed
    // const token = localStorage.getItem('token')
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`
    // }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    // Handle errors globally
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data)

      // You can handle specific error codes here
      switch (error.response.status) {
        case 401:
          // Unauthorized - redirect to login
          console.error('Unauthorized access')
          break
        case 404:
          console.error('Resource not found')
          break
        case 500:
          console.error('Server error')
          break
        default:
          console.error('An error occurred')
      }
    } else if (error.request) {
      // Request was made but no response received
      console.error('No response from server')
    } else {
      // Something else happened
      console.error('Error:', error.message)
    }

    return Promise.reject(error)
  }
)

// API endpoints
export const endpoints = {
  // Predictions
  getPredictions: (params) => api.get('/predictions', { params }),

  // Model data
  getModelData: (ticker) => api.get(`/models/${ticker}`),

  // Price data
  getPriceData: (ticker, minutes = 60) => api.get(`/prices/${ticker}`, {
    params: { minutes }
  }),

  // Settings
  getSettings: () => api.get('/settings'),
  updateSettings: (data) => api.post('/settings', data),
}

export default api
