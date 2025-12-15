import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useSettingsStore = create(
  persist(
    (set) => ({
      // Settings state
      probabilityThreshold: 30, // Default to 30% to show more predictions
      filterMode: 'all', // 'all', 'up', 'down'

      // Actions
      setProbabilityThreshold: (value) => set({ probabilityThreshold: value }),
      setFilterMode: (mode) => set({ filterMode: mode }),

      // Reset to defaults
      resetSettings: () => set({
        probabilityThreshold: 30,
        filterMode: 'all',
      }),
    }),
    {
      name: 'nasdaq-predictor-settings', // localStorage key
      version: 3, // Bump version to reset stored settings
    }
  )
)
