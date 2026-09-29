const modelSelect = document.querySelector('#model');
const statisticsDescription = document.querySelector('#statistics-description');
const metricCards = document.querySelector('#metric-cards');
const confusionMatrix = document.querySelector('#confusion-matrix');
const modelStatistics = new Map();

const titleCase = value => value.replace(/\b\w/g, character => character.toUpperCase());

function renderMatrix(statistics) {
  const { labels, confusion_matrix: matrix } = statistics;
  const maxCount = Math.max(...matrix.flat(), 1);
  confusionMatrix.replaceChildren();
  const caption = document.createElement('caption');
  caption.textContent = 'Confusion matrix — rows are actual emotions and columns are predictions.';
  confusionMatrix.append(caption);
  const header = document.createElement('tr');
  header.append(Object.assign(document.createElement('th'), { textContent: 'Actual / predicted', scope: 'col' }));
  labels.forEach(label => header.append(Object.assign(document.createElement('th'), { textContent: titleCase(label), scope: 'col' })));
  confusionMatrix.append(header);
  matrix.forEach((row, rowIndex) => {
    const tableRow = document.createElement('tr');
    tableRow.append(Object.assign(document.createElement('th'), { textContent: titleCase(labels[rowIndex]), scope: 'row' }));
    row.forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      cell.style.setProperty('--intensity', value / maxCount);
      tableRow.append(cell);
    });
    confusionMatrix.append(tableRow);
  });
}

function renderStatistics() {
  const statistics = modelStatistics.get(modelSelect.value);
  if (!statistics || !statistics.confusion_matrix) {
    statisticsDescription.textContent = 'Evaluation data is unavailable. Run python evaluate.py to generate it.';
    metricCards.replaceChildren();
    confusionMatrix.replaceChildren();
    return;
  }
  statisticsDescription.textContent = `${titleCase(statistics.architecture)} model · ${statistics.test_samples} clips from unseen test speakers ${statistics.speakers_test.join(', ')}.`;
  metricCards.innerHTML = `<article><span>Accuracy</span><strong>${(statistics.accuracy * 100).toFixed(1)}%</strong></article><article><span>Macro-F1</span><strong>${(statistics.macro_f1 * 100).toFixed(1)}%</strong></article><article><span>Test speakers</span><strong>${statistics.speakers_test.join(', ')}</strong></article>`;
  renderMatrix(statistics);
}

async function loadStatistics() {
  try {
    const response = await fetch('/models');
    if (!response.ok) throw new Error('Could not load model statistics.');
    const data = await response.json();
    data.models.forEach(({ architecture, statistics }) => modelStatistics.set(architecture, statistics));
    renderStatistics();
  } catch (error) {
    statisticsDescription.textContent = error.message;
  }
}

modelSelect.addEventListener('change', renderStatistics);
loadStatistics();
