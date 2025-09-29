<script setup lang="ts">

import { ref } from "vue";

const date = ref<string>("")
const specialist = ref<string>("")
const district = ref<string>("")
const comment = ref<string>("")
const $router = useRouter()

const districts = [
  "Центральный",
  "Северный",
  "Южный",
  "Восточный",
  "Западный"
]

const submitCheckup = () => {
  console.log({
    date: date.value,
    specialist: specialist.value,
    district: district.value,
    comment: comment.value
  })

  $router.push('/checkup/new2')
}
</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <v-card class="pa-4 rounded-lg" elevation="2">
      <v-card-title class="text-h5 font-weight-bold">
        Создать обследование
      </v-card-title>

      <v-card-text>
        <!-- Дата -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Дата обследования</label>
          <v-text-field v-model="date" type="date" prepend-inner-icon="ri-calendar-line" variant="outlined"
            density="comfortable" hide-details />
        </div>

        <!-- Специалист -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Специалист</label>
          <v-text-field v-model="specialist" prepend-inner-icon="ri-user-line" variant="outlined" density="comfortable"
            hide-details />
        </div>

        <!-- Район -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Район обследования</label>
          <v-select v-model="district" :items="districts" prepend-inner-icon="ri-map-pin-line" variant="outlined"
            density="comfortable" hide-details />
        </div>

        <!-- Комментарий -->
        <div class="mb-4">
          <label class="text-body-2 mb-1 d-block">Комментарий</label>
          <v-textarea v-model="comment" variant="outlined" rows="3" auto-grow hide-details />
        </div>

        <v-btn class="ma-auto" block color="primary" rounded="xl" @click="submitCheckup">
          Создать обследование
        </v-btn>
      </v-card-text>

    </v-card>
  </v-container>
</template>
