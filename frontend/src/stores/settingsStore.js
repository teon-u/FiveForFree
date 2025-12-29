import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useSettingsStore = create(
  persist(
    (set, get) => ({
      // Settings state
      probabilityThreshold: 0, // Default to 0% to show all predictions
      filterMode: 'all', // 'all', 'up', 'down'
      language: 'ko', // 'ko' or 'en' (default: Korean)
      theme: 'dark', // 'dark', 'light', 'system'

      // Actions
      setProbabilityThreshold: (value) => set({ probabilityThreshold: value }),
      setFilterMode: (mode) => set({ filterMode: mode }),
      setLanguage: (lang) => set({ language: lang }),
      setTheme: (theme) => set({ theme }),

      // Get effective theme (handles system preference)
      getEffectiveTheme: () => {
        const { theme } = get()
        if (theme === 'system') {
          return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
        }
        return theme
      },

      // Reset to defaults
      resetSettings: () => set({
        probabilityThreshold: 0,
        filterMode: 'all',
        language: 'ko',
        theme: 'dark',
      }),
    }),
    {
      name: 'nasdaq-predictor-settings', // localStorage key
      version: 6, // Bump version to reset stored settings
    }
  )
)
