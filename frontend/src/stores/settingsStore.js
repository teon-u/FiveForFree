import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useSettingsStore = create(
  persist(
    (set) => ({
      // Settings state
      probabilityThreshold: 0, // Default to 0% to show all predictions
      filterMode: 'all', // 'all', 'up', 'down'
      language: 'ko', // 'ko' or 'en' (default: Korean)

      // Actions
      setProbabilityThreshold: (value) => set({ probabilityThreshold: value }),
      setFilterMode: (mode) => set({ filterMode: mode }),
      setLanguage: (lang) => set({ language: lang }),

      // Reset to defaults
      resetSettings: () => set({
        probabilityThreshold: 0,
        filterMode: 'all',
        language: 'ko',
      }),
    }),
    {
      name: 'nasdaq-predictor-settings', // localStorage key
      version: 5, // Bump version to reset stored settings
    }
  )
)
