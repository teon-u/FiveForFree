import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useSettingsStore = create(
  persist(
    (set) => ({
      // Settings state
      targetPercent: 5.0,
      probabilityThreshold: 70,
      filterMode: 'all', // 'all', 'up', 'down'

      // Actions
      setTargetPercent: (value) => set({ targetPercent: value }),
      setProbabilityThreshold: (value) => set({ probabilityThreshold: value }),
      setFilterMode: (mode) => set({ filterMode: mode }),

      // Reset to defaults
      resetSettings: () => set({
        targetPercent: 5.0,
        probabilityThreshold: 70,
        filterMode: 'all',
      }),
    }),
    {
      name: 'nasdaq-predictor-settings', // localStorage key
      version: 1,
    }
  )
)
