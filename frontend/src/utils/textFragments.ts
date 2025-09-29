import { useApi } from '@/composables/useApi';

const $api = useApi()

export default {

  async getTextFragments(keys: string[]) {

    const pagesFragments = {} as any

    keys.forEach(key => {
      pagesFragments[key] = { title: '', content: '' }
    })

    const query = 'codes[]=' + keys.join('&codes[]=')
  
    await $api.get('/general/page-texts/?' + query)
    .then((response: any) => {
      response.data.forEach((item: any) => {
        if (item.content.includes('')) {
          const partnerFee = 10
          item.content = item.content.replaceAll('//PARTNER_FEE//', partnerFee)
            .replaceAll('//PARTNER_FEEx2//', 2 * partnerFee)
        }
        pagesFragments[item.code] = {
          title: item.title,
          content: item.content
        }
      })
    })

    return pagesFragments
  }

}
