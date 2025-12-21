import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useFilterStore = create(
  persist(
    (set, get) => ({
      // Direction filter (up/down)
      directions: ['up', 'down'],

      // Sector filter
      selectedSectors: [],
      selectedIndustries: [],

      // Probability filter
      probabilityPreset: 'all', // 'all', 'high', 'veryHigh', 'extreme', 'custom'
      probabilityRange: { min: 0, max: 100 },

      // Actions
      setDirections: (dirs) => set({ directions: dirs }),

      toggleDirection: (dir) => set((state) => ({
        directions: state.directions.includes(dir)
          ? state.directions.filter(d => d !== dir)
          : [...state.directions, dir]
      })),

      setSectors: (sectors) => set({ selectedSectors: sectors }),

      toggleSector: (sector) => set((state) => ({
        selectedSectors: state.selectedSectors.includes(sector)
          ? state.selectedSectors.filter(s => s !== sector)
          : [...state.selectedSectors, sector]
      })),

      setIndustries: (industries) => set({ selectedIndustries: industries }),

      toggleIndustry: (industry) => set((state) => ({
        selectedIndustries: state.selectedIndustries.includes(industry)
          ? state.selectedIndustries.filter(i => i !== industry)
          : [...state.selectedIndustries, industry]
      })),

      setProbabilityPreset: (preset) => {
        const presets = {
          all: { min: 0, max: 100 },
          high: { min: 70, max: 100 },
          veryHigh: { min: 80, max: 100 },
          extreme: { min: 90, max: 100 },
        }
        set({
          probabilityPreset: preset,
          probabilityRange: presets[preset] || { min: 0, max: 100 }
        })
      },

      setProbabilityRange: (range) => set({
        probabilityPreset: 'custom',
        probabilityRange: range
      }),

      // Reset filters
      resetFilters: () => set({
        directions: ['up', 'down'],
        selectedSectors: [],
        selectedIndustries: [],
        probabilityPreset: 'all',
        probabilityRange: { min: 0, max: 100 },
      }),

      // Get active filter count
      getActiveFilterCount: () => {
        const state = get()
        let count = 0
        if (state.directions.length < 2) count++
        if (state.selectedSectors.length > 0) count++
        if (state.selectedIndustries.length > 0) count++
        if (state.probabilityPreset !== 'all') count++
        return count
      },

      // Check if any filter is active
      hasActiveFilters: () => {
        const state = get()
        return (
          state.directions.length < 2 ||
          state.selectedSectors.length > 0 ||
          state.selectedIndustries.length > 0 ||
          state.probabilityPreset !== 'all'
        )
      }
    }),
    {
      name: 'nasdaq-predictor-filters',
      version: 1,
    }
  )
)
