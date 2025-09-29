<template>
  <v-menu
    v-model="menu"
    :close-on-content-click="false"
    transition="scale-transition"
    offset-y
    max-width="290"
    min-width="100"
  >
    <template #activator="{ props }">
      <v-text-field
        v-bind="props"
        v-model="displayDate"
        label="Выберите дату"
        readonly
        clearable
        density="compact"
        prepend-icon="mdi-calendar"
        style="width: 220px;"
      />
    </template>

    <v-date-picker
      v-model="selectedDate"
      locale="ru"
      hide-header
      @update:model-value="onDateSelected"
    />

  </v-menu>
</template>

<script lang="ts" setup>
import { format } from 'date-fns';
import { ru } from 'date-fns/locale';
import { computed, ref } from 'vue';

// date picker state
const menu = ref(false)
const selectedDate = ref<string | null>()

// Форматирование даты для отображения
const displayDate = computed(() =>
  selectedDate.value ? format(new Date(selectedDate.value), 'DD.MM.YYYY', { locale: ru }) : ''
)

// Автоматическое закрытие меню при выборе даты
const onDateSelected = (date: string | null) => {
  selectedDate.value = date
  menu.value = false
}
</script>

<style scoped>
.v-text-field {
  max-width: 220px;
}
</style>
