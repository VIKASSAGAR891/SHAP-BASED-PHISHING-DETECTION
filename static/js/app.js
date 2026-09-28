const form = document.getElementById('analysis-form');
const button = document.getElementById('analyse-btn');
const input = document.getElementById('url');
const emptyState = document.getElementById('empty-state');
const results = document.getElementById('results');
const activeModel = document.getElementById('active-model');
const modelOptions = document.querySelectorAll('.model-option');

const labels = {
  URLLength: 'URL length',
  DomainLength: 'Domain length',
  IsDomainIP: 'IP hostname',
  TLDLength: 'TLD length',
  NoOfSubDomain: 'Subdomains',
  HasObfuscation: 'Obfuscation',
  NoOfObfuscatedChar: 'Obfuscated chars',
  NoOfLettersInURL: 'Letters',
  LetterRatioInURL: 'Letter ratio',
  NoOfDegitsInURL: 'Digits',
  DegitRatioInURL: 'Digit ratio',
  NoOfEqualsInURL: 'Equals signs',
  NoOfQMarkInURL: 'Question marks',
  NoOfAmpersandInURL: 'Ampersands',
  NoOfOtherSpecialCharsInURL: 'Other special chars',
  SpacialCharRatioInURL: 'Special-char ratio',
  IsHTTPS: 'HTTPS',
  CharContinuationRate: 'Character continuation',
  NoOfURLRedirect: 'Redirect-like markers',
  NoOfSelfRedirect: 'Self-reference'
};


function setText(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = value;
  }
}


function formatValue(value) {
  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? value
      : value.toFixed(3);
  }

  return value;
}


function escapeHtml(value) {
  return String(value).replace(
    /[&<>'"]/g,
    character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[character])
  );
}


modelOptions.forEach(option => {
  option.addEventListener('click', async () => {
    modelOptions.forEach(item => {
      item.disabled = true;
    });

    try {
      const response = await fetch('/api/model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: option.dataset.model })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Unable to select the model.');
      }

      activeModel.textContent = data.model;
      modelOptions.forEach(item => {
        const selected = item.dataset.model === data.model;
        item.classList.toggle('selected', selected);
        item.setAttribute('aria-pressed', String(selected));
      });
    } catch (error) {
      showError(error.message || 'Unable to select the model.');
    } finally {
      modelOptions.forEach(item => {
        item.disabled = false;
      });
    }
  });
});


/*
 * Render the human-readable indicators returned by the API.
 *
 * If the backend returns the generic threshold message, we use
 * the actual SHAP explanation as a fallback so that the dashboard
 * never says "no anomaly" while the model has produced a strong
 * prediction.
 */
function renderIndicators(items, explanation, result) {
  const list = document.getElementById('indicators');

  list.innerHTML = '';

  let indicators = Array.isArray(items) ? [...items] : [];

  const genericMessage =
    'No high-level URL anomaly crossed the dashboard thresholds';

  if (
    indicators.length === 0 ||
    (indicators.length === 1 && indicators[0] === genericMessage)
  ) {
    indicators = (explanation || [])
      .filter(item => item.direction === 'supports prediction')
      .slice(0, 3)
      .map(item => `${item.feature} influenced the model prediction`);
  }

  if (indicators.length === 0) {
    indicators = [
      'The prediction was based on the combined URL feature pattern'
    ];
  }

  indicators.forEach(item => {
    const li = document.createElement('li');

    const mark = document.createElement('span');
    mark.className = 'indicator-mark';

    mark.textContent = '•';

    const text = document.createElement('span');
    text.textContent = item;

    li.appendChild(mark);
    li.appendChild(text);

    list.appendChild(li);
  });
}


/*
 * Render SHAP/model explanation.
 *
 * The API currently returns:
 *
 *   supports prediction
 *   opposes prediction
 *
 * We therefore colour the contribution according to whether
 * the feature supports or opposes the model's current prediction.
 */
function renderExplanations(items, result) {
  const box = document.getElementById('explanations');

  box.innerHTML = '';

  if (!Array.isArray(items) || items.length === 0) {
    box.innerHTML =
      '<div class="muted">SHAP explanation unavailable for this model build.</div>';
    return;
  }

  items.forEach(item => {
    const div = document.createElement('div');

    div.className = 'explanation-row';

    const main = document.createElement('div');
    main.className = 'explanation-main';

    const feature = document.createElement('div');
    feature.className = 'explanation-feature';
    feature.textContent = item.feature;

    const description = document.createElement('div');
    description.className = 'explanation-description';

    description.textContent =
      `${item.description} · value ${formatValue(item.raw_value)} · ${item.direction}`;

    main.appendChild(feature);
    main.appendChild(description);

    const impact = document.createElement('div');

    /*
     * "supports prediction" should use the same visual language
     * as the detected result.
     */
    const supportsPrediction =
      item.direction === 'supports prediction';

    if (supportsPrediction) {
      impact.className =
        'explanation-impact ' +
        (result === 'PHISHING'
          ? 'toward-phishing'
          : 'toward-legitimate');
    } else {
      impact.className =
        'explanation-impact ' +
        (result === 'PHISHING'
          ? 'toward-legitimate'
          : 'toward-phishing');
    }

    const numericImpact = Number(item.impact);

    impact.textContent =
      `${numericImpact >= 0 ? '+' : ''}${numericImpact.toFixed(3)}`;

    div.appendChild(main);
    div.appendChild(impact);

    box.appendChild(div);
  });
}


/*
 * Render extracted URL features.
 */
function renderFeatures(features) {
  const box = document.getElementById('feature-grid');

  box.innerHTML = '';

  Object.entries(features || {}).forEach(([key, value]) => {
    const cell = document.createElement('div');

    cell.className = 'feature-cell';

    const keyElement = document.createElement('div');
    keyElement.className = 'feature-key';
    keyElement.textContent = labels[key] || key;

    const valueElement = document.createElement('div');
    valueElement.className = 'feature-value';
    valueElement.textContent = formatValue(value);

    cell.appendChild(keyElement);
    cell.appendChild(valueElement);

    box.appendChild(cell);
  });
}


/*
 * Reset the error/empty-state styling before a new analysis.
 */
function resetEmptyState() {
  emptyState.classList.remove('error-state');
  emptyState.classList.remove('needs-training');
}


/*
 * Display an error message without breaking the dashboard.
 */
function showError(message) {
  results.classList.add('hidden');

  emptyState.classList.remove('hidden');
  emptyState.classList.add('error-state');

  emptyState.innerHTML = `
    <div class="empty-icon">!</div>
    <div>
      <div class="empty-title">ANALYSIS ERROR</div>
      <div class="empty-copy">${escapeHtml(message)}</div>
    </div>
  `;
}


/*
 * Main analysis request.
 */
form.addEventListener('submit', async event => {
  event.preventDefault();

  const url = input.value.trim();

  if (!url) {
    showError('Enter a URL to analyse.');
    input.focus();
    return;
  }

  button.disabled = true;

  button.innerHTML = `
    <span>ANALYSING</span>
    <span class="spinner"></span>
  `;

  resetEmptyState();

  try {
    const response = await fetch('/api/analyse', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        url: url
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || 'Analysis failed.'
      );
    }

    /*
     * Hide ready/error state and show actual results.
     */
    emptyState.classList.add('hidden');
    emptyState.classList.remove('error-state');

    results.classList.remove('hidden');

    /*
     * Result heading.
     */
    const isPhishing = data.result === 'PHISHING';

    setText(
      'result-title',
      isPhishing
        ? 'PHISHING DETECTED'
        : 'NO PHISHING SIGNAL DETECTED'
    );

    setText('result-url', data.url);

    setText(
      'result-badge',
      data.model_name
        ? data.model_name.toUpperCase()
        : 'MODEL'
    );

    /*
     * Risk/probability metrics.
     */
    setText(
      'risk-score',
      Number(data.risk_score).toFixed(1)
    );

    setText(
      'phishing-prob',
      Number(data.phishing_probability).toFixed(2)
    );

    setText(
      'legitimate-prob',
      Number(data.legitimate_probability).toFixed(2)
    );

    /*
     * Result colour.
     */
    const resultTitle =
      document.getElementById('result-title');

    resultTitle.className =
      isPhishing
        ? 'danger-title'
        : 'safe-title';

    /*
     * Risk meter.
     */
    const riskMeter =
      document.getElementById('risk-meter');

    const risk = Math.max(
      0,
      Math.min(100, Number(data.risk_score) || 0)
    );

    riskMeter.style.width = `${risk}%`;

    /*
     * Use the existing dashboard colour system.
     */
    riskMeter.style.background =
      isPhishing
        ? 'var(--danger)'
        : 'var(--accent)';

    /*
     * Render detailed information.
     */
    renderIndicators(
      data.indicators,
      data.explanation,
      data.result
    );

    renderExplanations(
      data.explanation,
      data.result
    );

    renderFeatures(
      data.features
    );

    /*
     * Bring the result section into view.
     */
    results.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });

  } catch (error) {

    showError(
      error.message || 'Unable to analyse the URL.'
    );

  } finally {

    button.disabled = false;

    button.innerHTML = `
      <span>ANALYSE</span>
      <span class="arrow">→</span>
    `;
  }
});