export const SECTORS = {
  technology: {
    name: { ko: '기술', en: 'Technology' },
    icon: '💻',
    industries: {
      software: { name: { ko: '소프트웨어', en: 'Software' } },
      semiconductors: { name: { ko: '반도체', en: 'Semiconductors' } },
      hardware: { name: { ko: '하드웨어', en: 'Hardware' } },
      internet: { name: { ko: '인터넷', en: 'Internet Services' } },
    }
  },
  healthcare: {
    name: { ko: '헬스케어', en: 'Healthcare' },
    icon: '🏥',
    industries: {
      biotech: { name: { ko: '바이오테크', en: 'Biotechnology' } },
      pharma: { name: { ko: '제약', en: 'Pharmaceuticals' } },
      devices: { name: { ko: '의료기기', en: 'Medical Devices' } },
    }
  },
  finance: {
    name: { ko: '금융', en: 'Finance' },
    icon: '🏦',
    industries: {
      banks: { name: { ko: '은행', en: 'Banks' } },
      insurance: { name: { ko: '보험', en: 'Insurance' } },
      fintech: { name: { ko: '핀테크', en: 'Fintech' } },
    }
  },
  consumer: {
    name: { ko: '소비재', en: 'Consumer' },
    icon: '🛒',
    industries: {
      ecommerce: { name: { ko: '이커머스', en: 'E-commerce' } },
      retail: { name: { ko: '소매', en: 'Retail' } },
      goods: { name: { ko: '소비재', en: 'Consumer Goods' } },
    }
  },
  energy: {
    name: { ko: '에너지', en: 'Energy' },
    icon: '⚡',
    industries: {
      oil: { name: { ko: '석유/가스', en: 'Oil & Gas' } },
      renewable: { name: { ko: '신재생', en: 'Renewable' } },
      utilities: { name: { ko: '유틸리티', en: 'Utilities' } },
    }
  },
  industrial: {
    name: { ko: '산업재', en: 'Industrial' },
    icon: '🏭',
    industries: {
      manufacturing: { name: { ko: '제조', en: 'Manufacturing' } },
      transportation: { name: { ko: '운송', en: 'Transportation' } },
      construction: { name: { ko: '건설', en: 'Construction' } },
    }
  },
  communication: {
    name: { ko: '통신', en: 'Communication' },
    icon: '📡',
    industries: {
      telecom: { name: { ko: '통신사', en: 'Telecom' } },
      media: { name: { ko: '미디어', en: 'Media' } },
      entertainment: { name: { ko: '엔터테인먼트', en: 'Entertainment' } },
    }
  },
}

// Ticker to sector/industry mapping (common NASDAQ stocks)
export const TICKER_SECTORS = {
  // Technology - Software
  MSFT: { sector: 'technology', industry: 'software' },
  ADBE: { sector: 'technology', industry: 'software' },
  CRM: { sector: 'technology', industry: 'software' },
  ORCL: { sector: 'technology', industry: 'software' },
  INTU: { sector: 'technology', industry: 'software' },
  NOW: { sector: 'technology', industry: 'software' },
  SNOW: { sector: 'technology', industry: 'software' },
  PANW: { sector: 'technology', industry: 'software' },
  CRWD: { sector: 'technology', industry: 'software' },
  ZS: { sector: 'technology', industry: 'software' },

  // Technology - Semiconductors
  NVDA: { sector: 'technology', industry: 'semiconductors' },
  AMD: { sector: 'technology', industry: 'semiconductors' },
  AVGO: { sector: 'technology', industry: 'semiconductors' },
  QCOM: { sector: 'technology', industry: 'semiconductors' },
  INTC: { sector: 'technology', industry: 'semiconductors' },
  MU: { sector: 'technology', industry: 'semiconductors' },
  MRVL: { sector: 'technology', industry: 'semiconductors' },
  LRCX: { sector: 'technology', industry: 'semiconductors' },
  KLAC: { sector: 'technology', industry: 'semiconductors' },
  AMAT: { sector: 'technology', industry: 'semiconductors' },

  // Technology - Hardware
  AAPL: { sector: 'technology', industry: 'hardware' },
  CSCO: { sector: 'technology', industry: 'hardware' },
  HPQ: { sector: 'technology', industry: 'hardware' },
  DELL: { sector: 'technology', industry: 'hardware' },

  // Technology - Internet
  GOOGL: { sector: 'technology', industry: 'internet' },
  GOOG: { sector: 'technology', industry: 'internet' },
  META: { sector: 'technology', industry: 'internet' },
  NFLX: { sector: 'technology', industry: 'internet' },
  ABNB: { sector: 'technology', industry: 'internet' },
  UBER: { sector: 'technology', industry: 'internet' },
  LYFT: { sector: 'technology', industry: 'internet' },
  SNAP: { sector: 'technology', industry: 'internet' },
  PINS: { sector: 'technology', industry: 'internet' },

  // Consumer - E-commerce
  AMZN: { sector: 'consumer', industry: 'ecommerce' },
  EBAY: { sector: 'consumer', industry: 'ecommerce' },
  ETSY: { sector: 'consumer', industry: 'ecommerce' },
  MELI: { sector: 'consumer', industry: 'ecommerce' },
  SE: { sector: 'consumer', industry: 'ecommerce' },
  SHOP: { sector: 'consumer', industry: 'ecommerce' },
  JD: { sector: 'consumer', industry: 'ecommerce' },
  PDD: { sector: 'consumer', industry: 'ecommerce' },
  BABA: { sector: 'consumer', industry: 'ecommerce' },

  // Consumer - Retail
  COST: { sector: 'consumer', industry: 'retail' },
  WMT: { sector: 'consumer', industry: 'retail' },
  TGT: { sector: 'consumer', industry: 'retail' },
  HD: { sector: 'consumer', industry: 'retail' },
  LOW: { sector: 'consumer', industry: 'retail' },
  DG: { sector: 'consumer', industry: 'retail' },
  DLTR: { sector: 'consumer', industry: 'retail' },

  // Consumer - Goods
  SBUX: { sector: 'consumer', industry: 'goods' },
  MCD: { sector: 'consumer', industry: 'goods' },
  NKE: { sector: 'consumer', industry: 'goods' },
  LULU: { sector: 'consumer', industry: 'goods' },
  PEP: { sector: 'consumer', industry: 'goods' },
  KO: { sector: 'consumer', industry: 'goods' },

  // Healthcare - Biotech
  MRNA: { sector: 'healthcare', industry: 'biotech' },
  GILD: { sector: 'healthcare', industry: 'biotech' },
  BIIB: { sector: 'healthcare', industry: 'biotech' },
  REGN: { sector: 'healthcare', industry: 'biotech' },
  VRTX: { sector: 'healthcare', industry: 'biotech' },
  ILMN: { sector: 'healthcare', industry: 'biotech' },
  SGEN: { sector: 'healthcare', industry: 'biotech' },

  // Healthcare - Pharma
  JNJ: { sector: 'healthcare', industry: 'pharma' },
  PFE: { sector: 'healthcare', industry: 'pharma' },
  LLY: { sector: 'healthcare', industry: 'pharma' },
  MRK: { sector: 'healthcare', industry: 'pharma' },
  ABBV: { sector: 'healthcare', industry: 'pharma' },
  BMY: { sector: 'healthcare', industry: 'pharma' },
  AZN: { sector: 'healthcare', industry: 'pharma' },

  // Healthcare - Medical Devices
  ISRG: { sector: 'healthcare', industry: 'devices' },
  DXCM: { sector: 'healthcare', industry: 'devices' },
  ALGN: { sector: 'healthcare', industry: 'devices' },
  IDXX: { sector: 'healthcare', industry: 'devices' },
  MDT: { sector: 'healthcare', industry: 'devices' },

  // Finance - Banks
  JPM: { sector: 'finance', industry: 'banks' },
  BAC: { sector: 'finance', industry: 'banks' },
  WFC: { sector: 'finance', industry: 'banks' },
  C: { sector: 'finance', industry: 'banks' },
  GS: { sector: 'finance', industry: 'banks' },
  MS: { sector: 'finance', industry: 'banks' },

  // Finance - Fintech
  PYPL: { sector: 'finance', industry: 'fintech' },
  SQ: { sector: 'finance', industry: 'fintech' },
  COIN: { sector: 'finance', industry: 'fintech' },
  SOFI: { sector: 'finance', industry: 'fintech' },
  AFRM: { sector: 'finance', industry: 'fintech' },
  V: { sector: 'finance', industry: 'fintech' },
  MA: { sector: 'finance', industry: 'fintech' },

  // Energy - Oil & Gas
  XOM: { sector: 'energy', industry: 'oil' },
  CVX: { sector: 'energy', industry: 'oil' },
  COP: { sector: 'energy', industry: 'oil' },
  EOG: { sector: 'energy', industry: 'oil' },
  OXY: { sector: 'energy', industry: 'oil' },

  // Energy - Renewable
  ENPH: { sector: 'energy', industry: 'renewable' },
  SEDG: { sector: 'energy', industry: 'renewable' },
  FSLR: { sector: 'energy', industry: 'renewable' },
  RUN: { sector: 'energy', industry: 'renewable' },

  // Industrial
  CAT: { sector: 'industrial', industry: 'manufacturing' },
  DE: { sector: 'industrial', industry: 'manufacturing' },
  HON: { sector: 'industrial', industry: 'manufacturing' },
  BA: { sector: 'industrial', industry: 'manufacturing' },
  GE: { sector: 'industrial', industry: 'manufacturing' },
  UPS: { sector: 'industrial', industry: 'transportation' },
  FDX: { sector: 'industrial', industry: 'transportation' },

  // Communication
  T: { sector: 'communication', industry: 'telecom' },
  VZ: { sector: 'communication', industry: 'telecom' },
  TMUS: { sector: 'communication', industry: 'telecom' },
  DIS: { sector: 'communication', industry: 'entertainment' },
  CMCSA: { sector: 'communication', industry: 'media' },
  WBD: { sector: 'communication', industry: 'media' },

  // EV
  TSLA: { sector: 'consumer', industry: 'goods' },
  RIVN: { sector: 'consumer', industry: 'goods' },
  LCID: { sector: 'consumer', industry: 'goods' },
  NIO: { sector: 'consumer', industry: 'goods' },
  LI: { sector: 'consumer', industry: 'goods' },
  XPEV: { sector: 'consumer', industry: 'goods' },
}

// Helper function to get sector info for a ticker
export const getTickerSector = (ticker) => {
  const mapping = TICKER_SECTORS[ticker]
  if (!mapping) return null

  const sector = SECTORS[mapping.sector]
  const industry = sector?.industries[mapping.industry]

  return {
    sectorCode: mapping.sector,
    industryCode: mapping.industry,
    sector: sector?.name,
    industry: industry?.name,
    icon: sector?.icon
  }
}

// Helper function to get all tickers in a sector
export const getTickersInSector = (sectorCode) => {
  return Object.entries(TICKER_SECTORS)
    .filter(([, mapping]) => mapping.sector === sectorCode)
    .map(([ticker]) => ticker)
}

// Helper function to get all tickers in an industry
export const getTickersInIndustry = (sectorCode, industryCode) => {
  return Object.entries(TICKER_SECTORS)
    .filter(([, mapping]) => mapping.sector === sectorCode && mapping.industry === industryCode)
    .map(([ticker]) => ticker)
}

// Get sector counts
export const getSectorCounts = (predictions) => {
  if (!predictions) return {}

  const counts = {}
  for (const p of predictions) {
    const mapping = TICKER_SECTORS[p.ticker]
    if (mapping) {
      if (!counts[mapping.sector]) {
        counts[mapping.sector] = { total: 0, industries: {} }
      }
      counts[mapping.sector].total++
      counts[mapping.sector].industries[mapping.industry] =
        (counts[mapping.sector].industries[mapping.industry] || 0) + 1
    }
  }
  return counts
}
