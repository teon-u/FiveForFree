import { useMemo } from 'react'
import clsx from 'clsx'
import { useNotificationStore } from '../stores/notificationStore'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'

// Format relative time
function formatRelativeTime(dateString, language) {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now - date
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHour = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHour / 24)

  if (language === 'ko') {
    if (diffSec < 60) return '방금 전'
    if (diffMin < 60) return `${diffMin}분 전`
    if (diffHour < 24) return `${diffHour}시간 전`
    if (diffDay < 7) return `${diffDay}일 전`
    return date.toLocaleDateString('ko-KR')
  }

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHour < 24) return `${diffHour}h ago`
  if (diffDay < 7) return `${diffDay}d ago`
  return date.toLocaleDateString('en-US')
}

// Group notifications by date
function groupByDate(notifications, language) {
  const groups = {}
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)

  notifications.forEach(notification => {
    const date = new Date(notification.time)
    let dateKey

    if (date.toDateString() === today.toDateString()) {
      dateKey = language === 'ko' ? '오늘' : 'Today'
    } else if (date.toDateString() === yesterday.toDateString()) {
      dateKey = language === 'ko' ? '어제' : 'Yesterday'
    } else {
      dateKey = date.toLocaleDateString(language === 'ko' ? 'ko-KR' : 'en-US')
    }

    if (!groups[dateKey]) {
      groups[dateKey] = []
    }
    groups[dateKey].push(notification)
  })

  return groups
}

function getNotificationIcon(type) {
  switch (type) {
    case 'signal_up': return '🟢'
    case 'signal_down': return '🔴'
    case 'discovery': return '🔥'
    case 'training_complete': return '✅'
    case 'watchlist': return '⭐'
    default: return '🔔'
  }
}

function NotificationItem({ notification, onRead, language }) {
  const { type, title, message, time, read } = notification

  return (
    <div
      className={clsx(
        'p-4 border-b border-surface-light cursor-pointer hover:bg-surface-light/50 transition-colors',
        !read && 'bg-blue-500/5'
      )}
      onClick={onRead}
      onKeyDown={(e) => e.key === 'Enter' && onRead()}
      role="button"
      tabIndex={0}
    >
      <div className="flex items-start gap-3">
        <span className="text-lg shrink-0">{getNotificationIcon(type)}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium text-sm truncate">{title}</span>
            <span className="text-xs text-gray-500 shrink-0">
              {formatRelativeTime(time, language)}
            </span>
          </div>
          <p className="text-sm text-gray-400 truncate">{message}</p>
        </div>
        {!read && (
          <div className="w-2 h-2 bg-blue-500 rounded-full shrink-0 mt-2" />
        )}
      </div>
    </div>
  )
}

export default function NotificationCenter({ onClose }) {
  const { language } = useSettingsStore()
  const { notifications, markAsRead, markAllAsRead, clearAll } = useNotificationStore()
  const tr = t(language)

  const groupedNotifications = useMemo(() => {
    return groupByDate(notifications, language)
  }, [notifications, language])

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') onClose()
  }

  return (
    <>
      <div
        className="modal-backdrop"
        onClick={onClose}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-label="Close"
      />

      <div
        className="fixed right-4 top-16 w-full max-w-[380px] max-h-[600px] bg-surface border border-surface-light rounded-xl shadow-2xl z-50 overflow-hidden flex flex-col"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-light shrink-0">
          <h2 className="font-bold flex items-center gap-2">
            🔔 {tr('notifications.title')}
          </h2>
          <div className="flex items-center gap-3">
            {notifications.length > 0 && (
              <>
                <button
                  onClick={markAllAsRead}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  {tr('notifications.markAllRead')}
                </button>
                <button
                  onClick={clearAll}
                  className="text-xs text-gray-400 hover:text-gray-300"
                >
                  {tr('notifications.clearAll')}
                </button>
              </>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Notification List */}
        <div className="flex-1 overflow-y-auto">
          {Object.entries(groupedNotifications).map(([date, items]) => (
            <div key={date}>
              <div className="px-4 py-2 text-xs text-gray-500 bg-surface-light/50 sticky top-0">
                {date}
              </div>
              {items.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onRead={() => markAsRead(notification.id)}
                  language={language}
                />
              ))}
            </div>
          ))}

          {notifications.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">🔔</div>
              <p>{tr('notifications.empty')}</p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
