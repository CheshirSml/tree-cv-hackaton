const localStorageKey = 'referal';
const expirationDays = 30;

export default {

  getReferralData () {
    const storedReferral = localStorage.getItem(localStorageKey);
    if (storedReferral) {
      try {
        return JSON.parse(storedReferral);
      } catch (e) {
        console.error('Ошибка разбора JSON:', e);
        return null;
      }
    }
    return null;
  },

  setReferralData (code: string, expiredDate: Date) {
    const referralData = {
        code: code,
        expired_date: expiredDate.toISOString(),
    };
    console.log('setReferralData', localStorageKey, referralData)
    localStorage.setItem(localStorageKey, JSON.stringify(referralData));
  },

  clearReferralData () {
    localStorage.removeItem(localStorageKey);
  },
  
  isReferralExpired (expiredDateString: string) {
    const expiredDate = new Date(expiredDateString)
    return expiredDate < new Date()
  },

  controlRefferalRecord (refCode: string|null|undefined) {
    let referralData = this.getReferralData()

    if(referralData){
      if(this.isReferralExpired(referralData.expired_date)){
        this.clearReferralData()
        referralData = null
      }
    }

    if (refCode && !referralData) {
        const currentDate = new Date();
        const expirationDate = new Date(currentDate);
        // TODO: стоит потом добавить на сервере проверку, что это валидный реферальный код
        expirationDate.setDate(currentDate.getDate() + expirationDays);
        this.setReferralData(refCode, expirationDate);
    }
  },

  getReferralCode () {
    const referralData = this.getReferralData()
    return referralData?.code || null
  },

}
