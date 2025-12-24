import { useState } from 'react'
import { useFilterStore } from '../../stores/filterStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { SECTORS } from '../../data/sectors'

const labels = {
  ko: {
    label: '섹터',
    all: '전체',
    clear: '초기화'
  },
  en: {
    label: 'Sector',
    all: 'All',
    clear: 'Clear'
  }
}

export default function SectorFilter() {
  const { language } = useSettingsStore()
  const { selectedSectors, toggleSector, setSectors } = useFilterStore()
  const [isOpen, setIsOpen] = useState(false)
  const tr = labels[language] || labels.ko

  const isAllSelected = selectedSectors.length === 0
  const sectorEntries = Object.entries(SECTORS)

  const handleAllClick = () => {
    setSectors([])
    setIsOpen(false)
  }

  const getSelectedLabel = () => {
    if (isAllSelected) return tr.all
    if (selectedSectors.length === 1) {
      const sector = SECTORS[selectedSectors[0]]
      return `${sector.icon} ${sector.name[language] || sector.name.ko}`
    }
    return `${selectedSectors.length}개 선택`
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-400">{tr.label}:</span>

        {/* Dropdown trigger */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-2
            ${!isAllSelected
              ? 'bg-purple-500/20 border border-purple-500 text-purple-400'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
        >
          {getSelectedLabel()}
          <svg
            className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Dropdown menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute top-full left-0 mt-2 w-56 bg-gray-800 rounded-lg shadow-xl border border-gray-700 z-20 overflow-hidden">
            {/* All option */}
            <button
              onClick={handleAllClick}
              className={`w-full px-4 py-2.5 text-left text-sm flex items-center gap-2 transition-colors
                ${isAllSelected
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
            >
              <span className="w-5 text-center">🌐</span>
              {tr.all}
            </button>

            <div className="border-t border-gray-700" />

            {/* Sector options */}
            {sectorEntries.map(([code, sector]) => {
              const isSelected = selectedSectors.includes(code)
              return (
                <button
                  key={code}
                  onClick={() => toggleSector(code)}
                  className={`w-full px-4 py-2.5 text-left text-sm flex items-center gap-2 transition-colors
                    ${isSelected
                      ? 'bg-purple-500/20 text-purple-400'
                      : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                    }`}
                >
                  <span className="w-5 text-center">{sector.icon}</span>
                  {sector.name[language] || sector.name.ko}
                  {isSelected && (
                    <svg className="w-4 h-4 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              )
            })}

            {/* Clear button */}
            {!isAllSelected && (
              <>
                <div className="border-t border-gray-700" />
                <button
                  onClick={handleAllClick}
                  className="w-full px-4 py-2 text-left text-sm text-gray-500 hover:text-white hover:bg-gray-700 transition-colors"
                >
                  {tr.clear}
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}
