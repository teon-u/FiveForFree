import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useNotificationStore = create(
  persist(
    (set, get) => ({
      // Permission state
      permission: 'default', // 'default' | 'granted' | 'denied'
      bannerDismissed: false,

      // Notification settings
      settings: {
        enabled: true,
        signalUp: true,
        signalDown: true,
        discovery: true,
        watchlist: true,
        trainingComplete: false,
        probabilityThreshold: 80,
        grades: ['A', 'B'],
        quietHours: {
          enabled: false,
          start: '22:00',
          end: '08:00',
        },
      },

      // Notification list
      notifications: [],

      // Actions
      requestPermission: async () => {
        if (!('Notification' in window)) return false

        const permission = await Notification.requestPermission()
        set({ permission })
        return permission === 'granted'
      },

      checkPermission: () => {
        if ('Notification' in window) {
          set({ permission: Notification.permission })
        }
      },

      dismissBanner: () => set({ bannerDismissed: true }),

      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),

      addNotification: (notification) => set((state) => ({
        notifications: [
          { ...notification, id: Date.now(), read: false, time: new Date().toISOString() },
          ...state.notifications
        ].slice(0, 100) // Keep max 100 notifications
      })),

      markAsRead: (id) => set((state) => ({
        notifications: state.notifications.map(n =>
          n.id === id ? { ...n, read: true } : n
        )
      })),

      markAllAsRead: () => set((state) => ({
        notifications: state.notifications.map(n => ({ ...n, read: true }))
      })),

      clearAll: () => set({ notifications: [] }),

      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id)
      })),

      // Check if in quiet hours
      isQuietHours: () => {
        const state = get()
        if (!state.settings.quietHours.enabled) return false

        const now = new Date()
        const hour = now.getHours()
        const minute = now.getMinutes()
        const currentTime = hour * 60 + minute

        const [startHour, startMin] = state.settings.quietHours.start.split(':').map(Number)
        const [endHour, endMin] = state.settings.quietHours.end.split(':').map(Number)
        const startTime = startHour * 60 + startMin
        const endTime = endHour * 60 + endMin

        // Handle overnight quiet hours (e.g., 22:00 - 08:00)
        if (startTime > endTime) {
          return currentTime >= startTime || currentTime < endTime
        }
        return currentTime >= startTime && currentTime < endTime
      },

      // Send notification
      sendNotification: (title, options = {}) => {
        const state = get()
        if (state.permission !== 'granted' || !state.settings.enabled) return false

        // Check quiet hours
        if (state.isQuietHours()) return false

        // Check notification type settings
        const { type } = options
        if (type === 'signal_up' && !state.settings.signalUp) return false
        if (type === 'signal_down' && !state.settings.signalDown) return false
        if (type === 'discovery' && !state.settings.discovery) return false
        if (type === 'watchlist' && !state.settings.watchlist) return false
        if (type === 'training_complete' && !state.settings.trainingComplete) return false

        try {
          // Browser notification
          new Notification(title, {
            icon: '/icons/icon-192.png',
            badge: '/icons/badge-72.png',
            tag: options.ticker || 'default',
            ...options
          })
        } catch {
          // Notification API might fail in some environments
          console.warn('Failed to send browser notification')
        }

        // Add to in-app notifications
        get().addNotification({
          type: options.type || 'default',
          title,
          message: options.body,
          ticker: options.ticker,
        })

        return true
      },

      getUnreadCount: () => {
        return get().notifications.filter(n => !n.read).length
      },
    }),
    {
      name: 'nasdaq-predictor-notifications',
      version: 1,
      partialize: (state) => ({
        bannerDismissed: state.bannerDismissed,
        settings: state.settings,
        notifications: state.notifications.slice(0, 50), // Only persist last 50
      }),
    }
  )
)
