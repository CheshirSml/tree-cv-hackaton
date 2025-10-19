<script setup lang="ts">

import AnnotationItem from '@/views/AnnotationItem.vue';

const props = defineProps({
  checkupData: {
    type: Object as () => any,
    required: true
  }
})

// вычисляем количество по состоянию
const counts = computed(() => {
  const healthy = props.checkupData.photos.filter((p: any) => p.annotation.condition === 'нормальное').length
  const unsatisfactory = props.checkupData.photos.filter((p: any) => p.annotation.condition === 'не удовлетворительное').length
  const critical = props.checkupData.photos.filter((p: any) => !['нормальное', 'не удовлетворительное'].includes(p.annotation.condition)).length

  return { healthy, unsatisfactory, critical }
})

</script>

<template>
  <!-- Обобщение -->
  <v-card class="pa-4 mb-6 rounded-lg" elevation="3">
    <div class="d-flex justify-space-around mb-4">
      <v-chip color="success" variant="flat" class="px-3">
        <strong>{{ counts.healthy }}</strong>&nbsp; здоровые
      </v-chip>

      <v-chip color="warning" variant="flat" class="px-3">
        <strong>{{ counts.unsatisfactory }}</strong>&nbsp; неуд.
      </v-chip>

      <v-chip color="error" variant="flat" class="px-3">
        <strong>{{ counts.critical }}</strong>&nbsp; аварийные
      </v-chip>
    </div>

    <v-divider />

    <div class="mt-4 text-center">
      <span class="text-subtitle-1 font-weight-medium text-medium-emphasis">
        Обследование <strong>{{ checkupData.photos.length }} деревьев</strong> завершено
      </span>
    </div>
  </v-card>

  <!-- Список деревьев -->
  <div v-for="tree in checkupData.photos" :key="tree.id" class="mb-6">
    <AnnotationItem v-if="tree.annotation" :tree="tree" :annotation="tree.annotation" />
  </div>
</template>
