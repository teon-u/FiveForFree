import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TickerCard from './TickerCard'
import { usePriceStore } from '../stores/priceStore'

// Mock the price store
vi.mock('../stores/priceStore', () => ({
  usePriceStore: vi.fn()
}))

// Mock the usePriceHistory hook (which includes useSparkline)
vi.mock('../hooks/usePriceHistory', () => ({
  useSparkline: vi.fn(() => ({
    data: { data: [100, 101, 102, 101, 103] },
    isLoading: false,
    error: null
  }))
}))

// Create a wrapper with QueryClientProvider
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
  function TestWrapper({ children }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    )
  }
  return TestWrapper
}

describe('TickerCard', () => {
  const defaultPrediction = {
    ticker: 'AAPL',
    probability: 75,
    direction: 'up',
    change_percent: 1.5,
    best_model: 'ensemble',
    hit_rate: 55,
    signal_rate: 0.12,
    practicality_grade: 'A',
  }

  const mockOnClick = vi.fn()
  const mockOnDetailClick = vi.fn()
  const mockClearPriceChanged = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    usePriceStore.mockImplementation((selector) => {
      const state = {
        prices: {},
        clearPriceChanged: mockClearPriceChanged,
      }
      return selector(state)
    })
  })

  it('renders ticker symbol correctly', () => {
    render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('AAPL')).toBeInTheDocument()
  })

  it('displays probability correctly', () => {
    const { container } = render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    // Probability is displayed in prob-badge div
    const probBadge = container.querySelector('.prob-badge')
    expect(probBadge).toBeInTheDocument()
    expect(probBadge.textContent).toContain('75')
  })

  it('shows direction indicator for upward prediction', () => {
    const { container } = render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    // Check for the up arrow in the prob-badge
    const probBadge = container.querySelector('.prob-badge')
    expect(probBadge).toBeInTheDocument()
    expect(probBadge.textContent).toContain('↑')
    expect(probBadge).toHaveClass('up')
  })

  it('displays practicality grade', () => {
    render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('A')).toBeInTheDocument()
  })

  it('displays formatted model name', () => {
    render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    // ensemble should be displayed as ENS
    expect(screen.getByText('ENS')).toBeInTheDocument()
  })

  it('shows hit rate as percentage', () => {
    render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText(/55/)).toBeInTheDocument()
  })

  it('applies correct style for A-grade prediction', () => {
    const highProbPrediction = {
      ...defaultPrediction,
      probability: 85,
      direction: 'up',
      practicality_grade: 'A',
    }

    const { container } = render(
      <TickerCard
        prediction={highProbPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    const card = container.querySelector('.ticker-card')
    expect(card).toHaveClass('grade-border-a')
    expect(card).toHaveClass('grade-a-animate')
  })

  it('applies correct style for B-grade prediction', () => {
    const downPrediction = {
      ...defaultPrediction,
      probability: 85,
      direction: 'down',
      practicality_grade: 'B',
    }

    const { container } = render(
      <TickerCard
        prediction={downPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    const card = container.querySelector('.ticker-card')
    expect(card).toHaveClass('grade-border-b')
  })

  it('uses real-time price from store when available', () => {
    usePriceStore.mockImplementation((selector) => {
      const state = {
        prices: {
          AAPL: {
            price: 195.50,
            change_percent: 2.3,
            priceChanged: false,
          }
        },
        clearPriceChanged: mockClearPriceChanged,
      }
      return selector(state)
    })

    render(
      <TickerCard
        prediction={defaultPrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    // Should display real-time price
    expect(screen.getByText('$195.50')).toBeInTheDocument()
  })

  it('renders C-grade style for C-grade prediction', () => {
    const cGradePrediction = {
      ...defaultPrediction,
      probability: 60,
      practicality_grade: 'C',
    }

    const { container } = render(
      <TickerCard
        prediction={cGradePrediction}
        onClick={mockOnClick}
        onDetailClick={mockOnDetailClick}
      />,
      { wrapper: createWrapper() }
    )

    const card = container.querySelector('.ticker-card')
    expect(card).toHaveClass('grade-border-c')
  })
})
