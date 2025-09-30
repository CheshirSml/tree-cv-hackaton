<script setup lang="ts">
defineProps({
  photoUrl: {
    type: String,
    default: ''
  },
  photoStatus: {
    type: String,
    required: true
  },
})
</script>


<template>
  <v-sheet class="pa-2 d-flex flex-column align-center" rounded="lg" elevation="2">
    <div class="position-relative w-100">
      <v-img v-if="photoUrl" :src="photoUrl" aspect-ratio="1" class="rounded-lg" cover />
      <!-- Лоадер поверх картинки -->
      <v-progress-circular v-if="photoStatus !== 'Готово'" indeterminate color="primary" size="32"
        class="position-absolute top-50 start-50 translate-middle" />
    </div>

    <!-- Статус -->
    <div class="d-flex align-center mt-2">
      <span class="me-2 rounded-circle" :style="{
        width: '10px',
        height: '10px',
        backgroundColor:
          photoStatus === 'Готово'
            ? '#4CAF50'
            : photoStatus === 'Ожидание'
              ? '#FFC107'
              : '#2196F3'
      }" />
      <span class="text-caption text-medium-emphasis">
        {{ photoStatus }}
      </span>
    </div>
  </v-sheet>
</template>
