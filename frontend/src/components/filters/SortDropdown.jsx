import { useState } from 'react'
import { useSortStore } from '../../stores/sortStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { SORT_OPTIONS, SORT_PRESETS } from '../../utils/sortUtils'

const labels = {
  ko: {
    sort: '정렬',
    presets: '프리셋',
    single: '단일 정렬',
    reset: '초기화'
  },
  en: {
    sort: 'Sort',
    presets: 'Presets',
    single: 'Single Sort',
    reset: 'Reset'
  }
}

export default function SortDropdown() {
  const [isOpen, setIsOpen] = useState(false)
  const { language } = useSettingsStore()
  const {
    sortMode,
    presetKey,
    singleSort,
    setPreset,
    setSingleSort,
    toggleSingleSortOrder,
    resetSort
  } = useSortStore()
  const tr = labels[language] || labels.ko

  const getCurrentLabel = () => {
    if (sortMode === 'preset') {
      return SORT_PRESETS[presetKey]?.label[language] || SORT_PRESETS[presetKey]?.label.ko
    }
    return SORT_OPTIONS[singleSort.field]?.label[language] || SORT_OPTIONS[singleSort.field]?.label.ko
  }

  const getCurrentOrder = () => {
    if (sortMode === 'preset') {
      return SORT_PRESETS[presetKey]?.configs[0]?.order || 'desc'
    }
    return singleSort.order
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
          bg-gray-700 text-gray-300 hover:bg-gray-600
          flex items-center gap-2"
      >
        <span className="text-gray-400">{tr.sort}:</span>
        <span>{getCurrentLabel()}</span>
        <span className="text-blue-400">
          {getCurrentOrder() === 'desc' ? '▼' : '▲'}
        </span>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full right-0 mt-2 w-72 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50">
            {/* Presets Section */}
            <div className="p-3 border-b border-gray-700">
              <div className="text-xs text-gray-400 mb-2">{tr.presets}</div>
              <div className="space-y-1">
                {Object.entries(SORT_PRESETS).map(([key, preset]) => (
                  <button
                    key={key}
                    onClick={() => {
                      setPreset(key)
                      setIsOpen(false)
                    }}
                    className={`
                      w-full text-left px-3 py-2 rounded-lg text-sm transition-colors
                      ${sortMode === 'preset' && presetKey === key
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'hover:bg-gray-700 text-gray-300'
                      }
                    `}
                  >
                    <div className="font-medium">
                      {preset.label[language] || preset.label.ko}
                    </div>
                    <div className="text-xs text-gray-500">
                      {preset.description[language] || preset.description.ko}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Single Sort Section */}
            <div className="p-3">
              <div className="text-xs text-gray-400 mb-2">{tr.single}</div>
              <div className="space-y-1">
                {Object.entries(SORT_OPTIONS).map(([key, option]) => (
                  <div
                    key={key}
                    className={`
                      flex items-center justify-between px-3 py-2 rounded-lg text-sm
                      ${sortMode === 'single' && singleSort.field === key
                        ? 'bg-blue-500/20'
                        : 'hover:bg-gray-700'
                      }
                    `}
                  >
                    <button
                      onClick={() => {
                        setSingleSort(key, option.defaultOrder)
                        setIsOpen(false)
                      }}
                      className="flex-1 text-left text-gray-300"
                    >
                      {option.label[language] || option.label.ko}
                    </button>
                    {sortMode === 'single' && singleSort.field === key && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleSingleSortOrder()
                        }}
                        className="px-2 py-1 text-blue-400 hover:bg-blue-500/20 rounded transition-colors"
                      >
                        {singleSort.order === 'desc' ? '▼' : '▲'}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Reset */}
            <div className="p-3 border-t border-gray-700">
              <button
                onClick={() => {
                  resetSort()
                  setIsOpen(false)
                }}
                className="w-full py-2 text-sm text-gray-400 hover:text-white transition-colors"
              >
                {tr.reset}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
