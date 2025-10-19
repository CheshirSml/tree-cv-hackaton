<script setup lang="ts">
import { useApi } from '@/composables/useApi';
import DetailsCheckupView from '@/views/DetailsCheckupView.vue';
import UploadPhotosView from '@/views/UploadPhotosView.vue';
import { ref } from "vue";
import { useRoute } from 'vue-router';

const $route = useRoute()
const checkupId = $route.params.id.toString()
const isLoading = ref(false)
const checkupData = ref<any | null>(null)

interface Tree {
  id: number
  photo: string
  title: string
  status: string
  defects: string[]
  dryBranches?: string
}

const $api = useApi()

const loadData = () => {
  isLoading.value = true
  $api.get('/checkups/' + checkupId + '/')
    .then((response) => {
      console.log('API response:', response)
      checkupData.value = response.data
      isLoading.value = false
    })
    .catch((error) => {
      console.error('API request failed:', error)
      // Покажет даже если WebView блокирует
      alert(
        `API request failed\nMessage: ${error.message}\n` +
        `${error.response ? JSON.stringify(error.response.data) : 'No response'}`
      )
      isLoading.value = false
    })

}

onMounted(() => loadData())
</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <!-- Заголовок участка -->
    <h2 class="text-h4 font-weight-bold mb-4">
      <IconBtn icon="ri-arrow-left-line" class="mr-1" to="/" />
      Участок: {{ checkupData?.area_detail?.title }}
    </h2>

    <template v-if="checkupData">
      <UploadPhotosView v-if="checkupData.status == 'pending'" v-model:checkup-data="checkupData" />
      <DetailsCheckupView v-else :checkup-data="checkupData" />
    </template>

  </v-container>
</template>
