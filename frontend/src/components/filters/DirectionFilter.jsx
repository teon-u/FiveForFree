import { useFilterStore } from '../../stores/filterStore'
import { useSettingsStore } from '../../stores/settingsStore'

const labels = {
  ko: { up: '상승', down: '하락' },
  en: { up: 'Up', down: 'Down' }
}

export default function DirectionFilter() {
  const { language } = useSettingsStore()
  const { directions, toggleDirection } = useFilterStore()
  const tr = labels[language] || labels.ko

  return (
    <div className="flex gap-1">
      <button
        onClick={() => toggleDirection('up')}
        className={`
          px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
          flex items-center gap-1.5
          ${directions.includes('up')
            ? 'bg-green-600 text-white shadow-md shadow-green-500/30'
            : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200'
          }
        `}
      >
        <span>📈</span>
        {tr.up}
      </button>
      <button
        onClick={() => toggleDirection('down')}
        className={`
          px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
          flex items-center gap-1.5
          ${directions.includes('down')
            ? 'bg-red-600 text-white shadow-md shadow-red-500/30'
            : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200'
          }
        `}
      >
        <span>📉</span>
        {tr.down}
      </button>
    </div>
  )
}
