import { useState } from 'react'
import { useFilterStore } from '../../stores/filterStore'
import { useSettingsStore } from '../../stores/settingsStore'

const labels = {
  ko: {
    title: '확률 필터',
    all: '전체',
    high: 'High (70%+)',
    veryHigh: 'Very High (80%+)',
    extreme: 'Extreme (90%+)',
    custom: '커스텀',
    apply: '적용',
    cancel: '취소',
    min: '최소',
    max: '최대'
  },
  en: {
    title: 'Probability Filter',
    all: 'All',
    high: 'High (70%+)',
    veryHigh: 'Very High (80%+)',
    extreme: 'Extreme (90%+)',
    custom: 'Custom',
    apply: 'Apply',
    cancel: 'Cancel',
    min: 'Min',
    max: 'Max'
  }
}

export default function ProbabilityFilter() {
  const [isOpen, setIsOpen] = useState(false)
  const { language } = useSettingsStore()
  const { probabilityPreset, probabilityRange, setProbabilityPreset, setProbabilityRange } = useFilterStore()
  const tr = labels[language] || labels.ko

  const presets = [
    { key: 'all', label: tr.all },
    { key: 'high', label: tr.high },
    { key: 'veryHigh', label: tr.veryHigh },
    { key: 'extreme', label: tr.extreme },
  ]

  const getCurrentLabel = () => {
    if (probabilityPreset === 'custom') {
      return `${probabilityRange.min}%-${probabilityRange.max}%`
    }
    const preset = presets.find(p => p.key === probabilityPreset)
    return preset?.label || tr.all
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
          flex items-center gap-2
          ${probabilityPreset !== 'all'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }
        `}
      >
        <span>📊</span>
        {getCurrentLabel()}
        <span className="text-xs opacity-75">▼</span>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 mt-2 w-56 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50">
            <div className="p-3">
              <div className="text-xs text-gray-400 mb-2">{tr.title}</div>

              {/* Presets */}
              <div className="space-y-1">
                {presets.map(preset => (
                  <button
                    key={preset.key}
                    onClick={() => {
                      setProbabilityPreset(preset.key)
                      setIsOpen(false)
                    }}
                    className={`
                      w-full text-left px-3 py-2 rounded-lg text-sm transition-colors
                      ${probabilityPreset === preset.key
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'hover:bg-gray-700 text-gray-300'
                      }
                    `}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Custom Range */}
              <div className="mt-3 pt-3 border-t border-gray-700">
                <button
                  onClick={() => {
                    if (probabilityPreset !== 'custom') {
                      setProbabilityRange({ min: 60, max: 100 })
                    }
                  }}
                  className={`
                    w-full text-left px-3 py-2 rounded-lg text-sm transition-colors
                    ${probabilityPreset === 'custom'
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'hover:bg-gray-700 text-gray-300'
                    }
                  `}
                >
                  {tr.custom}
                </button>

                {probabilityPreset === 'custom' && (
                  <div className="mt-2 px-3 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-8">{tr.min}</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={probabilityRange.min}
                        onChange={(e) => setProbabilityRange({
                          ...probabilityRange,
                          min: Math.min(parseInt(e.target.value), probabilityRange.max - 5)
                        })}
                        className="flex-1"
                      />
                      <span className="text-xs text-gray-300 w-10">{probabilityRange.min}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-8">{tr.max}</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={probabilityRange.max}
                        onChange={(e) => setProbabilityRange({
                          ...probabilityRange,
                          max: Math.max(parseInt(e.target.value), probabilityRange.min + 5)
                        })}
                        className="flex-1"
                      />
                      <span className="text-xs text-gray-300 w-10">{probabilityRange.max}%</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
