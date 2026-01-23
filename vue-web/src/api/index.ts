import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosResponse } from 'axios'
import { ElMessage, ElNotification } from 'element-plus'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_TIMEOUT = 60000 // 60 seconds for long-running operations

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    
    if (error.response?.status === 401) {
      ElMessage.error('Session expired. Please login again.')
      // Handle logout
    } else if (error.response?.status === 500) {
      ElNotification({
        title: 'Server Error',
        message: message,
        type: 'error',
        duration: 5000
      })
    } else {
      ElMessage.error(message)
    }
    
    return Promise.reject(error)
  }
)

// Generic request function
async function request<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.request<T>(config)
  return response.data
}

// API Methods
export const api = {
  // GET request
  get: <T>(url: string, params?: Record<string, any>): Promise<T> => {
    return request<T>({ method: 'GET', url, params })
  },
  
  // POST request
  post: <T>(url: string, data?: any): Promise<T> => {
    return request<T>({ method: 'POST', url, data })
  },
  
  // PUT request
  put: <T>(url: string, data?: any): Promise<T> => {
    return request<T>({ method: 'PUT', url, data })
  },
  
  // DELETE request
  delete: <T>(url: string): Promise<T> => {
    return request<T>({ method: 'DELETE', url })
  },
  
  // Upload file
  upload: <T>(url: string, formData: FormData, onProgress?: (percent: number) => void): Promise<T> => {
    return request<T>({
      method: 'POST',
      url,
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percent)
        }
      }
    })
  },
  
  // Download file
  download: async (url: string, filename: string): Promise<void> => {
    const response = await apiClient.get(url, {
      responseType: 'blob'
    })
    
    const blob = new Blob([response.data])
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.click()
    URL.revokeObjectURL(link.href)
  }
}

// Export individual methods and client
export { apiClient, API_BASE_URL }
export default api
