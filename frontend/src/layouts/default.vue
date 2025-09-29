<script lang="ts" setup>

import logoPhoto from '@images/logo.png';
import { ref } from "vue";
import { useRouter } from "vue-router";

const isLoading = ref(false)
const router = useRouter()

router.beforeEach((to, from, next) => {
  isLoading.value = true
  next()
})

router.afterEach(() => {
  isLoading.value = false
})
</script>

<template>
  <VProgressLinear v-if="isLoading" indeterminate color="success" height="5" class="fixed top-0 left-0 w-full z-50"
    style="position: fixed !important; z-index: 1100;" />
  <v-container class="pa-4" style="max-width: 480px">
    <div class="d-flex justify-space-between pb-1">
      <VAvatar :size="40" rounded>
        <VImg :src="logoPhoto" />
      </VAvatar>

      <h2 class="text-h3 text-center">ЭкоКонтроль</h2>

      <IconBtn>
        <VIcon icon="ri-settings-5-line" />
      </IconBtn>

    </div>
  </v-container>

  <RouterView />
</template>
