<template>
  <VDialog
    v-model="isDialogVisible"
    persistent
    :class="sizeClass"
    z-index="10010"
  >
    <DialogCloseBtn @click="isDialogVisible = false; emit('hide');" />

    <VCard :title="title">
      <template v-slot:title>
        <div v-if="title" class="v-card-item__content">
          <div class="v-card-title">{{ title }}</div>
        </div>
        <slot name="title" />
      </template>
      <VCardText>
        <slot></slot>
      </VCardText>

      <VCardText v-if="showFooter" class="d-flex justify-end gap-3 flex-wrap">
        <VBtn v-if="showCancel"
          @click="close"
          color="secondary"
          variant="tonal"
        >
          {{ cancelLabel }}
        </VBtn>
        <VBtn v-if="showSave"
          @click="save"
          :disabled="disabled"
        >
          {{ saveLabel }}
        </VBtn>
      </VCardText>
    </VCard>
  </VDialog>
</template>

<script setup lang="ts">
  
  type Size = 'sm' | 'md' | 'lg' | 'xl';

  const emit = defineEmits(['hide', 'save']);

  const isDialogVisible = ref(false)

  const props = defineProps({
    title: {
      type: String,
      default: 'Dialog title'
    },
    showCancel: {
      type: Boolean,
      default: true
    },
    cancelLabel: {
      type: String,
      default: 'Отмена'
    },
    showSave: {
      type: Boolean,
      default: true
    },
    saveLabel: {
      type: String,
      default: 'Подтвердить'
    },
    backgroundClosing: {
      type: Boolean,
      default: true
    },
    showFooter: {
      type: Boolean,
      default: true
    },
    disabled: {
      type: Boolean,
      default: false
    },
    size: {
      type: String as () => Size,
      default: 'sm'
    }
  })

  const sizeClass = computed(() => {
    switch(props.size) {
      case 'sm': return 'v-dialog-sm';
      case 'md': return 'v-dialog-md';
      case 'lg': return 'v-dialog-lg';
      case 'xl': return 'v-dialog-xl';
      default: 'v-dialog-sm';
    }
  })

  const open = () => {
    isDialogVisible.value = true
  }

  const close = () => {
    emit('hide')
    if (!props.backgroundClosing) {
      hide()
    }
  }

  const hide = () => {
    isDialogVisible.value = false
  }

  const save = () => {
    emit('save')
  }

  defineExpose({
    open,
    close,
    hide,
  })

</script>

