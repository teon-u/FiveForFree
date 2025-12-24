import DirectionFilter from './DirectionFilter'
import ProbabilityFilter from './ProbabilityFilter'
import GradeFilter from './GradeFilter'
import SectorFilter from './SectorFilter'
import SortDropdown from './SortDropdown'
import { useFilterStore } from '../../stores/filterStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { SECTORS } from '../../data/sectors'

const labels = {
  ko: {
    reset: '초기화',
    activeFilters: '필터 적용됨'
  },
  en: {
    reset: 'Reset',
    activeFilters: 'Filters Active'
  }
}

export default function FilterBar() {
  const { language } = useSettingsStore()
  const { resetFilters, hasActiveFilters, getActiveFilterCount, directions, probabilityPreset, selectedGrades, selectedSectors } = useFilterStore()
  const tr = labels[language] || labels.ko

  const filterCount = getActiveFilterCount()
  const showReset = hasActiveFilters()

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-3">
      <div className="flex flex-wrap items-center gap-3">
        {/* Direction Filter */}
        <DirectionFilter />

        {/* Divider */}
        <div className="w-px h-6 bg-gray-700" />

        {/* Probability Filter */}
        <ProbabilityFilter />

        {/* Divider */}
        <div className="w-px h-6 bg-gray-700" />

        {/* Grade Filter */}
        <GradeFilter />

        {/* Divider */}
        <div className="w-px h-6 bg-gray-700" />

        {/* Sector Filter */}
        <SectorFilter />

        {/* Divider */}
        <div className="w-px h-6 bg-gray-700" />

        {/* Sort Dropdown */}
        <SortDropdown />

        {/* Spacer */}
        <div className="flex-1" />

        {/* Active Filter Badge & Reset */}
        {showReset && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 bg-gray-700 px-2 py-1 rounded">
              {filterCount} {tr.activeFilters}
            </span>
            <button
              onClick={resetFilters}
              className="text-sm text-gray-400 hover:text-white transition-colors
                flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-700"
            >
              <span>✕</span>
              {tr.reset}
            </button>
          </div>
        )}
      </div>

      {/* Active Filter Tags (for visibility) */}
      {showReset && (
        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-gray-700/50">
          {directions.length === 1 && (
            <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full flex items-center gap-1">
              {directions[0] === 'up' ? '📈 상승' : '📉 하락'}
            </span>
          )}
          {probabilityPreset !== 'all' && (
            <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full">
              확률 {probabilityPreset === 'custom' ? '커스텀' : `${probabilityPreset === 'high' ? '70%+' : probabilityPreset === 'veryHigh' ? '80%+' : '90%+'}`}
            </span>
          )}
          {selectedGrades.length > 0 && (
            <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
              등급: {selectedGrades.join(', ')}
            </span>
          )}
          {selectedSectors.length > 0 && (
            <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-1 rounded-full flex items-center gap-1">
              {selectedSectors.map(s => SECTORS[s]?.icon).join(' ')}
              {selectedSectors.length === 1
                ? SECTORS[selectedSectors[0]]?.name[language] || SECTORS[selectedSectors[0]]?.name.ko
                : `${selectedSectors.length}개 섹터`}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
