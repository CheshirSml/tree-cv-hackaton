import axios from 'axios'

export const useApi = () => {
  const baseURL = import.meta.env.VITE_BASE_URL

  const instance = axios.create({
    baseURL,
    headers: {}
  })

  // перехватчик ответа
  instance.interceptors.response.use(
    response => response,
    error => {
      // покажет тип ошибки и сообщение
      alert(`API Error:\n${error.message}\n${error.response ? JSON.stringify(error.response.data) : ''}`)
      console.error('API Error:', error)
      return Promise.reject(error)
    }
  )

  return instance
}
