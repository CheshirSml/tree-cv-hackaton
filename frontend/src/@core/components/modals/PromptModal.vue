<template>
  <BaseModal ref="confirmModal"
    :title="label"
    saveLabel="Подтвердить"
    :backgroundClosing="false"
    :disabled="!localValue"
    @hide="hide"
    @save="finish"
  >
    <div tabindex="0"
      @keypress.enter="finish"
    >
      <VTextField v-model="localValue"
        label="Сумма:"
        :options="options"
        :autofocus="true"
      />
    </div>
  </BaseModal>
</template>

<script lang="ts">

import BaseModal from '@/@core/components/modals/BaseModal.vue';
import { defineComponent } from "vue";

  export default defineComponent({
    
    components: {
      BaseModal
    },

    data() {
      return {
        label: 'Please, input value:',
        localValue: '',
        options: [] as Array<string>,
        resolvePromise: null as null | any,
        rejectPromise: null as null | any,
      }
    },
    methods: {
      async open(label: string) {
        this.label = label
        this.localValue = ''
        const confirmModal = this.$refs.confirmModal as any
        confirmModal.open()
        return new Promise((resolve, reject) => {
          this.resolvePromise = resolve
          this.rejectPromise = reject
        })
      },

      hide() {
        if (this.resolvePromise) {
          this.resolvePromise('')
          const confirmModal = this.$refs.confirmModal as any
          confirmModal.hide()
        }
      },

      finish() {
        if (this.resolvePromise) {
          this.resolvePromise(this.localValue)
          const confirmModal = this.$refs.confirmModal as any
          confirmModal.hide()
        }
      }
    },

    computed: {
      numberRule() {
        return (v: string) => {
          if (!v.trim()) return true;
          if (!isNaN(parseFloat(v)) && parseFloat(v) > 0) return true;
          return 'Non-zero number required';
        }
      },

    }
  })

</script>

<style>
.mb-32 {
  margin-bottom: 128px;
}
</style>
