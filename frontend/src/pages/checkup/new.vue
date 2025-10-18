<script setup lang="ts">

import { useApi } from '@/composables/useApi';
import { ref } from "vue";
import { toast } from 'vue3-toastify';

const $api = useApi()
const $router = useRouter()

const isLoading = ref(false)
const isProcessing = ref(false)
const checkupData = ref<any | null>(null)
const districtAreas = ref<any[] | null>(null)

const loadData = async () => {
  isLoading.value = true

  await $api.get('/areas/')
    .then((response) => {
      districtAreas.value = response.data
    })

  await $api.get('/checkups/prototype/')
    .then((response) => {
      checkupData.value = response.data
    })

  isLoading.value = false
}

const submitCheckup = () => {
  isProcessing.value = true
  $api.post('/checkups/', checkupData.value)
    .then((response: any) => {
      isProcessing.value = false
      $router.push('/checkup/' + response.data.id + '/')
    })
    .catch(() => {
      isProcessing.value = false
      toast.error("Не удалось создать оьследование. ", {
        autoClose: 8000,
      })
    })

}

onMounted(() => loadData())

</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <v-card class="pa-4 rounded-lg" elevation="2">

      <VProgressLinear v-if="isLoading" indeterminate absolute />

      <v-card-title class="text-h5 font-weight-bold">
        Создать обследование
      </v-card-title>

      <v-card-text v-if="!!checkupData">
        <!-- Дата -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Дата обследования</label>
          <v-text-field v-model="checkupData.report_date" type="date" prepend-inner-icon="ri-calendar-line"
            variant="outlined" density="comfortable" hide-details :disabled="isProcessing" />
        </div>

        <!-- Специалист -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Специалист</label>
          <v-text-field model-value="Иванов Петр Сергеевич" prepend-inner-icon="ri-user-line" variant="outlined"
            density="comfortable" hide-details readonly :disabled="isProcessing" />
        </div>

        <!-- Район -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Район обследования</label>
          <v-select v-model="checkupData.area" :items="districtAreas ?? []" prepend-inner-icon="ri-map-pin-line"
            variant="outlined" density="comfortable" hide-details item-title="title" item-value="id"
            :disabled="isProcessing || !districtAreas" :loading="!districtAreas" />
        </div>

        <!-- Комментарий -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Комментарий</label>
          <v-textarea v-model="checkupData.comment" variant="outlined" :disabled="isProcessing" rows="3" auto-grow
            hide-details />
        </div>

        <v-btn class="ma-auto" block color="primary" :disabled="isProcessing" rounded="xl" @click="submitCheckup">
          Создать обследование
        </v-btn>
      </v-card-text>

    </v-card>
  </v-container>
</template>
