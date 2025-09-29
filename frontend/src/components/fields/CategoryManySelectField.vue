<template>
  <Multiselect v-model="localValues"
    v-bind="categoryMultiselect"
    :placeholder="placeholder"
    track-by="title"
    value-prop="id"
    label="title"
    mode="tags"
    class="one-line"
    :searchable="true"
    :append-new-option="false"
    :append-new-tag="false"
    :add-tag-on="[]"
    :add-option-on="[]"
    name="get_service"
    :loading="!categories"
    @update:model-value="changeValue"
  >
    <template v-slot:tag="{ option, handleTagRemove, disabled }">
      <div
        class="multiselect-tag is-user"
        :class="{
          'is-disabled': disabled
        }"
      >
        <img :src="option.micon" class="character-label-icon">
        {{ option.title }}
        <span
          v-if="!disabled"
          class="multiselect-tag-remove"
          @click="handleTagRemove(option, $event)"
        >
          <span class="multiselect-tag-remove-icon"></span>
        </span>
      </div>
    </template>
  
    <template v-slot:option="{ option }">
      <img class="character-option-icon" style="" :src="option.micon"> {{ option.title }}
    </template>
  </Multiselect>
</template>

<script lang="ts" setup>
	import { useUserStore } from "@/store";
import Multiselect from '@vueform/multiselect';

  const $emit = defineEmits()
  const $store = useUserStore()
  const categories = ref<Array<any>|null>(null)
  const localValues = ref<Array<number>>([])

  const $props = defineProps({
    modelValue: {
      type: Array as () => Array<number>,
      required: true
    },
    placeholder: {
      type: String,
      default: ''
    }
  })

  const categoryItems = computed(() => {
    const result = []
    if (!categories.value) {
      return []
    }
    return categories.value
    // if (localValues.value.length >= 2) {
    //   const v1 = categories.value.find(x => true || x.id == localValues.value[0])
    //   const v2 = v1 ? v1.childrens.find((x: any) => true || x.id == localValues.value[1]) : null
    //   if (v1) {
    //     result.push(v1)
    //   }
    //   if (v2) {
    //     result.push(v2)
    //   }
    //   return result
    // }
    // else if (localValues.value.length == 1) {
    //   const v = categories.value.find(x => x.id == localValues.value[0])
    //   if (!v) {
    //     return categories.value
    //   }
    //   result.push(v)
    //   const category = categories.value.find(x => x.id == localValues.value[0])
    //   result.push(...(category.childrens))
    // } else {
    //   result.push(...(categories.value))
    // }
    // return result
  })

  const categoryMultiselect = computed(() => {
    return {
      mode: 'tags',
      // closeOnSelect: localValues.value.length >= 1,
      closeOnSelect: false,
      options: categoryItems.value,
      searchable: true,
      createOption: true,
    }
  })

  const changeValue = () => {
    const value = localValues.value.slice()
    $emit('update:modelValue', value)
  }

  watch(() => $props.modelValue, () => {
    localValues.value = $props.modelValue
  })

  onMounted(async () => {
    categories.value = await $store.getCategories()
    localValues.value = $props.modelValue
  })

</script>

<style>
  :root {
    --ms-tag-bg: #8c57ff;
  }
  .multiselect.one-line {
    border: solid 1px #8c57ff;
    border-radius: 6px;
    padding-top: 4px;
    padding-bottom: 4px;
  }
  .two-items .multiselect-tag:first-child {
    /* display: none; */
  }
  .one-line .multiselect-tags-search-wrapper {
    position: absolute;
  }

  .character-option-icon {
    margin: 0 6px 0 0;
    height: 22px;
  }
  
  .character-label-icon {
    margin: 0 6px 0 0;
    height: 26px;
  }

</style>
