import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import OverviewTab from './OverviewTab'
import PerformanceTab from './PerformanceTab'
import EnsembleTab from './EnsembleTab'
import FinancialTab from './FinancialTab'

export default function ModelDetailModal({ ticker, onClose }) {
  const [activeTab, setActiveTab] = useState('overview')

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [onClose])

  // Fetch overview data
  const { data: overviewData, isLoading: overviewLoading } = useQuery({
    queryKey: ['model-overview', ticker],
    queryFn: async () => {
      const response = await api.get(`/models/${ticker}/overview`)
      return response.data
    },
    enabled: !!ticker && activeTab === 'overview',
    staleTime: 60000,
  })

  // Fetch performance data
  const { data: performanceData, isLoading: performanceLoading } = useQuery({
    queryKey: ['model-performance', ticker],
    queryFn: async () => {
      const response = await api.get(`/models/${ticker}/performance`)
      return response.data
    },
    enabled: !!ticker && activeTab === 'performance',
    staleTime: 60000,
  })

  // Fetch ensemble data
  const { data: ensembleData, isLoading: ensembleLoading } = useQuery({
    queryKey: ['model-ensemble', ticker],
    queryFn: async () => {
      const response = await api.get(`/models/${ticker}/ensemble`)
      return response.data
    },
    enabled: !!ticker && activeTab === 'ensemble',
    staleTime: 60000,
  })

  // Fetch financial data
  const { data: financialData, isLoading: financialLoading } = useQuery({
    queryKey: ['model-financial', ticker],
    queryFn: async () => {
      const response = await api.get(`/models/${ticker}/financial`)
      return response.data
    },
    enabled: !!ticker && activeTab === 'financial',
    staleTime: 60000,
  })

  const isLoading = activeTab === 'overview' ? overviewLoading :
                     activeTab === 'performance' ? performanceLoading :
                     activeTab === 'ensemble' ? ensembleLoading :
                     financialLoading

  const tabs = [
    { id: 'overview', label: '📊 Overview', icon: '📊' },
    { id: 'performance', label: '📈 Performance', icon: '📈' },
    { id: 'ensemble', label: '🎭 Ensemble', icon: '🎭' },
    { id: 'financial', label: '💰 Financial', icon: '💰' },
  ]

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 overflow-y-auto">
        <div className="flex items-center justify-center min-h-screen p-4">
          <div
            className="bg-surface rounded-xl shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 rounded-t-xl z-10">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white">{ticker}</h2>
                  <p className="text-sm text-gray-400 mt-1">
                    Model Performance Analysis
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="text-gray-400 hover:text-white transition-colors p-2"
                  aria-label="Close"
                >
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {/* Tabs */}
              <div className="flex gap-2 mt-4">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      px-4 py-2 rounded-lg font-medium transition-all
                      ${
                        activeTab === tab.id
                          ? 'bg-primary text-white shadow-lg'
                          : 'bg-surface-light text-gray-400 hover:text-white hover:bg-surface-lighter'
                      }
                    `}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {isLoading ? (
                <div className="flex items-center justify-center min-h-[400px]">
                  <div className="text-center">
                    <div className="spinner mb-4" />
                    <p className="text-gray-400">Loading model data...</p>
                  </div>
                </div>
              ) : (
                <>
                  {activeTab === 'overview' && overviewData && (
                    <OverviewTab data={overviewData} ticker={ticker} />
                  )}
                  {activeTab === 'performance' && performanceData && (
                    <PerformanceTab data={performanceData} ticker={ticker} />
                  )}
                  {activeTab === 'ensemble' && ensembleData && (
                    <EnsembleTab data={ensembleData} ticker={ticker} />
                  )}
                  {activeTab === 'financial' && financialData && (
                    <FinancialTab data={financialData} ticker={ticker} />
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
