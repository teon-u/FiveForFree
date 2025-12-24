import { useFilterStore } from '../../stores/filterStore'
import { useSettingsStore } from '../../stores/settingsStore'

const GRADES = ['A', 'B', 'C', 'D']

const gradeColors = {
  A: { bg: 'bg-green-500/20', border: 'border-green-500', text: 'text-green-400', hover: 'hover:bg-green-500/30' },
  B: { bg: 'bg-blue-500/20', border: 'border-blue-500', text: 'text-blue-400', hover: 'hover:bg-blue-500/30' },
  C: { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400', hover: 'hover:bg-yellow-500/30' },
  D: { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400', hover: 'hover:bg-red-500/30' },
}

const labels = {
  ko: {
    label: '등급',
    all: '전체',
    tooltip: {
      A: '매우 높음 (Precision ≥80%)',
      B: '높음 (Precision ≥70%)',
      C: '보통 (Precision ≥60%)',
      D: '낮음 (Precision <60%)',
    }
  },
  en: {
    label: 'Grade',
    all: 'All',
    tooltip: {
      A: 'Very High (Precision ≥80%)',
      B: 'High (Precision ≥70%)',
      C: 'Medium (Precision ≥60%)',
      D: 'Low (Precision <60%)',
    }
  }
}

export default function GradeFilter() {
  const { language } = useSettingsStore()
  const { selectedGrades, toggleGrade, setGrades } = useFilterStore()
  const tr = labels[language] || labels.ko

  const isAllSelected = selectedGrades.length === 0

  const handleAllClick = () => {
    setGrades([])
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-400">{tr.label}:</span>

      {/* All button */}
      <button
        onClick={handleAllClick}
        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all
          ${isAllSelected
            ? 'bg-gray-600 text-white'
            : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
      >
        {tr.all}
      </button>

      {/* Grade buttons */}
      {GRADES.map((grade) => {
        const isSelected = selectedGrades.includes(grade)
        const colors = gradeColors[grade]

        return (
          <button
            key={grade}
            onClick={() => toggleGrade(grade)}
            title={tr.tooltip[grade]}
            className={`px-3 py-1.5 rounded-lg text-sm font-bold transition-all border
              ${isSelected
                ? `${colors.bg} ${colors.border} ${colors.text}`
                : `bg-gray-800 border-gray-700 text-gray-400 ${colors.hover}`
              }`}
          >
            {grade}
          </button>
        )
      })}
    </div>
  )
}
