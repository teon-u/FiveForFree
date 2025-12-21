import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useWatchlistStore = create(
  persist(
    (set, get) => ({
      watchlist: [],

      addTicker: (ticker) => set((state) => {
        if (state.watchlist.find(w => w.ticker === ticker)) return state
        return {
          watchlist: [...state.watchlist, {
            ticker,
            alertEnabled: true,
            addedAt: new Date().toISOString()
          }]
        }
      }),

      removeTicker: (ticker) => set((state) => ({
        watchlist: state.watchlist.filter(w => w.ticker !== ticker)
      })),

      toggleAlert: (ticker) => set((state) => ({
        watchlist: state.watchlist.map(w =>
          w.ticker === ticker ? { ...w, alertEnabled: !w.alertEnabled } : w
        )
      })),

      setAlertEnabled: (ticker, enabled) => set((state) => ({
        watchlist: state.watchlist.map(w =>
          w.ticker === ticker ? { ...w, alertEnabled: enabled } : w
        )
      })),

      isInWatchlist: (ticker) => {
        return get().watchlist.some(w => w.ticker === ticker)
      },

      getAlertEnabled: (ticker) => {
        const item = get().watchlist.find(w => w.ticker === ticker)
        return item?.alertEnabled ?? false
      },

      reorder: (fromIndex, toIndex) => set((state) => {
        const newList = [...state.watchlist]
        const [removed] = newList.splice(fromIndex, 1)
        newList.splice(toIndex, 0, removed)
        return { watchlist: newList }
      }),

      getWatchlistTickers: () => {
        return get().watchlist.map(w => w.ticker)
      },

      clearAll: () => set({ watchlist: [] }),
    }),
    {
      name: 'nasdaq-predictor-watchlist',
      version: 1,
    }
  )
)
