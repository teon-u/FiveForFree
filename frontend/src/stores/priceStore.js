import { create } from 'zustand'

export const usePriceStore = create((set, get) => ({
  // Real-time prices: { ticker: { price, change_percent, timestamp, updated } }
  prices: {},

  // Connection status
  isConnected: false,
  lastUpdate: null,

  // Actions
  setConnected: (connected) => set({ isConnected: connected }),

  // Update multiple prices at once (from WebSocket price_update)
  updatePrices: (priceUpdates, timestamp) => {
    const currentPrices = get().prices
    const newPrices = { ...currentPrices }

    priceUpdates.forEach(({ ticker, price, change_percent }) => {
      const prevPrice = currentPrices[ticker]?.price
      newPrices[ticker] = {
        price,
        change_percent,
        timestamp,
        // Track if price changed for flash animation
        priceChanged: prevPrice !== undefined && prevPrice !== price,
        priceDirection: prevPrice !== undefined ? (price > prevPrice ? 'up' : price < prevPrice ? 'down' : null) : null,
        updated: Date.now(),
      }
    })

    set({ prices: newPrices, lastUpdate: timestamp })
  },

  // Update single price
  updatePrice: (ticker, price, change_percent) => {
    const currentPrices = get().prices
    const prevPrice = currentPrices[ticker]?.price

    set({
      prices: {
        ...currentPrices,
        [ticker]: {
          price,
          change_percent,
          timestamp: new Date().toISOString(),
          priceChanged: prevPrice !== undefined && prevPrice !== price,
          priceDirection: prevPrice !== undefined ? (price > prevPrice ? 'up' : price < prevPrice ? 'down' : null) : null,
          updated: Date.now(),
        },
      },
    })
  },

  // Get price for specific ticker
  getPrice: (ticker) => get().prices[ticker],

  // Clear flash animation flag for a ticker
  clearPriceChanged: (ticker) => {
    const currentPrices = get().prices
    if (currentPrices[ticker]) {
      set({
        prices: {
          ...currentPrices,
          [ticker]: {
            ...currentPrices[ticker],
            priceChanged: false,
            priceDirection: null,
          },
        },
      })
    }
  },
}))
