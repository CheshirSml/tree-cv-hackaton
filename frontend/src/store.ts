import { useApi } from '@/composables/useApi'
import { defineStore } from 'pinia'
const $api = useApi()

export const useUserStore = defineStore('user', {
  state: () => ({
    account: null as any|null,
    categories: null as Array<any>|null,
    cart: null as Array<any>|null,
    notifications: null as Array<any>|null,
    workCityId: null as number|null
  }),
  getters: {
    currentAccount: (state: any) => state.account,
    currentCart: (state: any) => state.cart,
    currentNotifications: (state: any) => state.notifications,
    currentWorkCityId: (state: any) => state.account && state.account.profile ? state.account.profile.city : null,
  },
  actions: {

    async loginByToken() {
      await $api.get('/auth/profile/')
        .then((response: any) => {
          this.account = response.data
          console.log('Получил аккаунт ', this.account)
        })
        .catch(() => {
          this.account = null
        })
        return this.account
    },
    
    async logout() {
      await $api.get('/auth/logout/')
        .then(() => {
          this.account = null
          localStorage.setItem('token', '');
        })
        .catch(() => {
          this.account = null
          alert('Ошибка')
        })
    },
    
    async getCategories() {
      await $api.get('/categories/')
        .then((response) => {
          this.categories = response.data
        })
        .catch(() => {
          this.categories = []
        })
      return this.categories
    },

    async loadCart() {
      if (!this.account) {
        this.cart = null
        return
      }
      await $api.get('/account/cart/')
        .then((response) => {
          this.cart = response.data
        })
        .catch(() => {
          this.cart = null
        })
    },

    updateProfile(profile: any) {
      this.account.profile = profile
    },

    async loadNotifications() {
      if (!this.account) {
        this.notifications = null
        return
      }
      await $api.get('/account/notifications/')
        .then((response) => {
          this.notifications = response.data.results
        })
        .catch(() => {
          // this.notifications = null
        })
    },
    
    setWorkCity(workCityId: number|null) {
      this.workCityId = workCityId
    },
  },
})
