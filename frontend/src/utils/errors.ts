export default {
  getErrorText(error : any) {
    if (error && error.response && error.response.data) {
      let errorData = error.response.data
      const isErrorDataIsObject = typeof errorData === 'object' && !Array.isArray(errorData) && errorData !== null
      if (isErrorDataIsObject && 'non_field_errors' in errorData) {
        errorData = errorData.non_field_errors
      }
      if (
        Array.isArray(errorData) 
        && errorData.length > 0 
        && typeof errorData[0] === 'string'
      ) {
        return errorData[0]
      } else if (typeof errorData === 'string') {
        if (errorData.length > 255) {
          return ''
        }
        return errorData
      }
      if (isErrorDataIsObject) {
        let message = ''
        for (let property in errorData) {
          if (Array.isArray(errorData[property])) {
            message += errorData[property].join(', ')
          }
        }
        if (message) {
          return message
        }
      }
    }
    console.log('error', error)
    return ''
  },
}
