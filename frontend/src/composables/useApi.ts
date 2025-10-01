import axios from 'axios';

export const useApi = () => {
  const baseURL = import.meta.env.VITE_BASE_URL

  const instance = axios.create({
    baseURL,
    headers: { }
  })

  // Set the AUTH token for any request
  instance.interceptors.request.use(function (config) {
    const token = localStorage.getItem('token');
    // console.log('token', token)
    config.headers.Authorization =  token ? `Token ${token}` : '';
    return config;
  })

  return instance
}

