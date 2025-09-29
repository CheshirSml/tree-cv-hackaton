<script setup lang="ts">
import emptyProfilePhoto from '@images/avatars/empty-profile-photo.png';

const investigations = ref([
  {
    title: 'Участок: А-12, Сокольники',
    created_date: '12.09.2025',
    count: 12,
    status: 'Завершено',
    color: 'success'
  },
  {
    title: 'Участок: А-12, Сокольники',
    created_date: '12.09.2025',
    count: 10,
    status: 'Обработка 65%',
    color: 'primary'
  },
  {
    title: 'Участок: А-12, Сокольники',
    created_date: '12.09.2025',
    count: 12,
    status: 'Ожидание',
    color: 'warning'
  },
])

</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <VRow class="mt-2">
      <VCol cols="12">
        <VCard class="pa-4 rounded-lg">
          <VCardText class="text-center pt-6">
            <VAvatar :size="100" color="primary" variant="tonal">
              <VImg :src="emptyProfilePhoto" />
            </VAvatar>

            <h5 class="text-h5 mt-4">
              Иванов Петр Сергеевич
            </h5>

            <div class="mt-1">Северное лесничество</div>

            <VChip label color="primary" size="small" class="text-capitalize mt-2">
              Дендролог
            </VChip>
          </VCardText>
        </VCard>
      </VCol>

      <VCol cols="12">
        <VBtn class="ma-auto" block color="primary" rounded="xl" to="/checkup/new">
          Создать обследование
        </VBtn>
      </VCol>

      <VCol cols="12" class="mt-4">
        <h2 class="text-h4 text-center">Мои обследования</h2>
      </VCol>

      <VCol v-for="investigation in investigations" cols="12">
        <VAlert :title="investigation.title" border="start" :border-color="investigation.color"
          style="box-shadow: 0px 4px 4px #ccc;">

          <ul>
            <li>Дата: {{ investigation.created_date }}</li>
            <li>Количество фото: {{ investigation.count }}</li>
          </ul>

          <VChip :color="investigation.color" style="position: absolute; right: 10px; bottom: 11px; ">{{
            investigation.status }}</VChip>
        </VAlert>
      </VCol>

      <VCol cols="12" class="my-4">
        <VPagination />
      </VCol>

    </VRow>
  </v-container>
</template>
