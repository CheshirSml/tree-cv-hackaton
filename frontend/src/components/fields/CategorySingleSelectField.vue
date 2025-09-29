<template>
  <!-- <span>{{ localValues }}</span> -->
  <!-- <span>{{ modelValue }}</span> -->
  <!-- <span>{{ categories }}</span> -->
  <div class="position-relative " :class="{ 'big-search': isBig }">
    <Multiselect v-model="localValues"
      v-bind="categoryMultiselect"
      :placeholder="placeholder"
      track-by="title"
      value-prop="id"
      label="title"
      mode="tags"
      class="one-line"
      :class="{ 'two-items': localValues.length > 1, 'is-invalid': errorMessage }"
      :append-new-option="false"
      :append-new-tag="false"
      :add-tag-on="[]"
      :add-option-on="[]"
      :name="name"
      :searchable="false"
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
      <template v-slot:multiplelabel="{ values }">
        <div>deded</div>
      </template>
    </Multiselect>
    <VIcon v-if="isBig" icon="ri-search-line" style="position: absolute; top: calc(50% - 1px); left: 17px; transform: translateY(-50%);" color="primary" />
    <div v-if="errorMessage" class="error-message">
      {{ errorMessage }}
    </div>
  </div>
</template>

<script lang="ts" setup>
	import { useUserStore } from "@/store";
import Multiselect from '@vueform/multiselect';
import '@vueform/multiselect/themes/default.css';

  const $emit = defineEmits()
  const $store = useUserStore()
  const categories = ref<Array<any>|null>(null)
  const localValues = ref<Array<number>>([])
  type ValidationRule = (value: any) => string | boolean
  const errorMessage = ref<string | null>(null)

  const $props = defineProps({
    modelValue: {
      type: Number as () => number|null,
      required: true
    },
    placeholder: {
      type: String,
      default: ''
    },
    isBig: {
      type: Boolean,
      default: false
    },
    name: {
      type: String,
      default: ''
    },
    rules: {
      type: Array as () => ValidationRule[],
      default: () => []
    },
  })

  const categoryItems = computed(() => {
    const result = []
    if (localValues.value.length >= 2) {
      const v1 = categories.value?.find(x => x.id == localValues.value[0])
      const v2 = v1 ? v1.childrens.find((x: any) => x.id == localValues.value[1]) : null
      if (v1) {
        result.push(v1)
      }
      if (v2) {
        result.push(v2)
      }
      return result
    }
    else if (localValues.value.length == 1) {
      const v = categories.value?.find((x: any) => x.id == localValues.value[0])
      if (!v) {
        return categories.value??[]
      }
      result.push(v)
      const category = categories.value?.find(x => x.id == localValues.value[0])
      result.push(...(category.childrens))
    } else {
      result.push(...(categories.value??[]))
    }
    return result
  })
  const categoryMultiselect = computed(() => {
    return {
      mode: 'tags',
      // closeOnSelect: localValues.value.length >= 1,
      closeOnSelect: localValues.value.length >= 0,
      // options: categoryItems.value,
      options: categories.value as Array<any>,
      searchable: true,
      createOption: true,
    }
  })

  const categoriesPathById = (id: number|null) => {
    if (!id || !categories.value) {
      return []
    }

    for (let i = 0; i < categories.value.length; i++) {
      const category = categories.value[i]
      if (category.id == id) {
        return [category.id]
      }
      for (let l = 0; l < category.childrens.length; l++) {
        const children = category.childrens[l]
        if (children.id == id) {
          return [category.id, children.id]
        }
      }
    }

    return []
  }

  const changeValue = () => {
    const value = localValues.value.length ? localValues.value[localValues.value.length - 1] : null
    $emit('update:modelValue', value)
  }

  watch(() => $props.modelValue, () => {
    localValues.value = categoriesPathById($props.modelValue)
  })

  const validate = () => {
    errorMessage.value = null
    for (const rule of $props.rules) {
      const result = rule(localValues.value)
      if (typeof result === 'string') {
        errorMessage.value = result
        break
      }
    }
  }

  watch(localValues, validate)

  onMounted(async () => {
    const localCategories = await $store.getCategories() as any[]
    categories.value = localCategories.map((x: any) => {
      return {
        id: x.id,
        title: x.title,
        icon: x.icon,
        micon: x.micon,
        childrens: []
      }
    })
    localValues.value = categoriesPathById($props.modelValue)
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

  .big-search .multiselect {
    padding-top: 8px;
    padding-bottom: 8px;
    padding-left: 35px;

  }

  .big-search .multiselect .multiselect-tag {
    padding-bottom: .325rem;
  }

  /* .big-search .multiselect .multiselect-wrapper {
    padding-left: 30px;
  } */

</style>
<style scoped>
.is-invalid {
  border-color: rgb(var(--v-theme-error)) !important;
}
.error-message {
  color: rgb(var(--v-theme-error));
  font-size: 0.75rem;
  margin-top: 4px;
}
</style>
