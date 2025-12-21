import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { SORT_PRESETS } from '../utils/sortUtils'

export const useSortStore = create(
  persist(
    (set, get) => ({
      // Current sort settings
      sortMode: 'preset', // 'preset' | 'single' | 'multi'
      presetKey: 'bestOpportunity',
      singleSort: {
        field: 'probability',
        order: 'desc'
      },
      multiSort: [
        { field: 'grade', order: 'desc' },
        { field: 'probability', order: 'desc' }
      ],

      // Actions
      setPreset: (presetKey) => set({
        sortMode: 'preset',
        presetKey
      }),

      setSingleSort: (field, order) => set({
        sortMode: 'single',
        singleSort: { field, order }
      }),

      toggleSingleSortOrder: () => set((state) => ({
        singleSort: {
          ...state.singleSort,
          order: state.singleSort.order === 'desc' ? 'asc' : 'desc'
        }
      })),

      setMultiSort: (configs) => set({
        sortMode: 'multi',
        multiSort: configs
      }),

      addMultiSort: (field, order) => set((state) => ({
        sortMode: 'multi',
        multiSort: [...state.multiSort, { field, order }].slice(0, 3) // Max 3
      })),

      removeMultiSort: (index) => set((state) => ({
        multiSort: state.multiSort.filter((_, i) => i !== index)
      })),

      // Get current sort configs
      getCurrentSortConfigs: () => {
        const state = get()
        switch (state.sortMode) {
          case 'preset':
            return SORT_PRESETS[state.presetKey]?.configs || []
          case 'single':
            return [state.singleSort]
          case 'multi':
            return state.multiSort
          default:
            return [{ field: 'probability', order: 'desc' }]
        }
      },

      // Reset to default
      resetSort: () => set({
        sortMode: 'preset',
        presetKey: 'bestOpportunity',
        singleSort: { field: 'probability', order: 'desc' },
        multiSort: []
      })
    }),
    {
      name: 'nasdaq-predictor-sort',
      version: 1,
    }
  )
)
