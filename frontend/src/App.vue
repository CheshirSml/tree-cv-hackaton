<template>
  <VApp>
    <div v-if="isReadyForWork">
      <RouterView />
    </div>
    <div v-else id="loading-bg">
      <div class="loading">
        <div class="effect-1 effects"></div>
        <div class="effect-2 effects"></div>
        <div class="effect-3 effects"></div>
      </div>
    </div>
  </VApp>
</template>

<script setup lang="ts">

import { useUserStore } from "@/store";
import { useRoute } from 'vue-router';
import 'vue3-toastify/dist/index.css';

const $store = useUserStore()
const $route = useRoute()
const isReadyForWork = ref(false)

const currentAccount = computed(() => {
  return $store.currentAccount
})

onMounted(async () => {
  isReadyForWork.value = true
})

</script>
