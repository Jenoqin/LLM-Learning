// 使用实际页面函数检查跨章节状态回归；浏览器布局仍需实机验证。
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const elements = new Map(), handlers = {}, writes = new Map();
function element(key) {
  if (!elements.has(key)) {
    const classes = new Set();
    elements.set(key, {textContent:'', innerHTML:'', children:[], style:{}, attrs:{}, dataset:{},
      setAttribute(k,v){this.attrs[k]=String(v);}, addEventListener(){}, focus(){}, scrollIntoView(){},
      classList:{add(k){classes.add(k);},remove(k){classes.delete(k);},contains(k){return classes.has(k);},toggle(k,on){if(on??!classes.has(k)){classes.add(k);return true;}classes.delete(k);return false;}}});
  }
  return elements.get(key);
}
const tabs=['loss','lr'].map(mode=>{const e=element('tab-'+mode);e.dataset.chart=mode;return e;});
const anchors=['start','boundary','end'].map(anchor=>{const e=element('anchor-'+anchor);e.dataset.anchor=anchor;return e;});
const context=vm.createContext({console,location:{hash:'#map'},localStorage:{getItem(){return null;},setItem(k,v){writes.set(k,v);}},
  setTimeout(){},clearTimeout(){},document:{querySelector:element,querySelectorAll(s){return s==='[data-chart]'?tabs:s==='[data-anchor]'?anchors:[];},addEventListener(type,fn){handlers[type]=fn;}},
  window:{addEventListener(){},scrollTo(){},matchMedia(){return {matches:false,addEventListener(){}};}}});
const core=fs.readFileSync('dist/core.mjs','utf8').replaceAll('export ','');
const app=fs.readFileSync('dist/app.js','utf8').replace(/^import[^\n]+\n/,'');
vm.runInContext(core+'\n'+app,context);
context.training=JSON.parse(fs.readFileSync('dist/assets/training-data.json','utf8'));
const run=code=>vm.runInContext(code,context);
function click({id='',dataset={},attribute}={}){handlers.click({target:{closest(s){return s==='button'?{id,dataset,hasAttribute(name){return name===attribute;}}:null;}}});}
run("records=training;chartMode='lr';location.hash='#evidence';render();");
assert.equal(element('tab-lr').attrs['aria-pressed'],'true');
assert.equal(element('tab-loss').attrs['aria-pressed'],'false');
assert(element('#training-chart').attrs['aria-label'].startsWith('学习率'));
run("location.hash='#terms';render();location.hash='#evidence';render();");
assert.equal(element('tab-lr').attrs['aria-pressed'],'true');
assert(element('#training-chart').attrs['aria-label'].startsWith('学习率'));
click({dataset:{anchor:'boundary'}});
assert.equal(run('records.points[recordIndex].epoch'),2);
assert.equal(element('tab-loss').attrs['aria-pressed'],'true');
click({dataset:{anchor:'end'}});assert.equal(run('records.points[recordIndex].global'),79390);
// 只通过自测：不得宣称六节都已完成。
run("location.hash='#quiz';answers=Object.fromEntries(questions.map((q,i)=>[i,q.answer]));submitted=true;render();");
click({attribute:'data-complete'});
assert.equal(element('#progress-text').textContent,'已完成 1 / 6');
assert(element('#toast').textContent.startsWith('自测已通过；'));
// 错题重做：保留正确回答，删除错题回答，仍需完成错题才能通过。
run('answers[1]=0;submitted=true;render();');
click({id:'quiz-wrong'});
assert.equal(run('quizLocked.size'),5);assert.equal(run('answers[1]'),undefined);
assert.equal(run('submitted'),false);
assert(element('#main').innerHTML.includes('name="question-0" value="0" checked disabled'));
assert(element('#main').innerHTML.includes('name="question-1" value="0"   required'));
click({attribute:'data-complete'});
assert(element('#toast').textContent.includes('先完成六道题'));
console.log('通过：跨章节曲线状态、三处记录定位、自测与章节完成分离、只重做错题。');
