import tensorflow as tf


class F1_score(tf.keras.metrics.Metric):# {{{
  def __init__(self, thresholds=0.5, name='f1', **kwargs):
    super(F1_score, self).__init__(name=name, **kwargs)
    self.f1 = self.add_weight(name='f1', initializer='zeros')
    self.tp = self.add_weight(name='tp', initializer='zeros')
    self.fp = self.add_weight(name='fp', initializer='zeros')
    self.fn = self.add_weight(name='fn', initializer='zeros')
    self.thresholds=thresholds

  def update_state(self, y_true, y_pred, sample_weight=None):
    min_delta=1e-6
    y_pred=tf.cast(tf.where(y_pred>self.thresholds,1,0),tf.int8)
    y_true=tf.cast(y_true,tf.int8)

    tp=tf.math.count_nonzero(y_pred*y_true,dtype=tf.float32)
    fp=tf.math.count_nonzero(y_pred*(1-y_true),dtype=tf.float32)
    fn=tf.math.count_nonzero((1-y_pred)*y_true,dtype=tf.float32)

    self.tp.assign_add(tp)
    self.fp.assign_add(fp)
    self.fn.assign_add(fn)

    self.f1.assign(2*self.tp/(2*self.tp+self.fp+self.fn+min_delta))

  def result(self):
    return self.f1

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.f1.assign(0.)
    self.tp.assign(0.)
    self.fp.assign(0.)
    self.fn.assign(0.)
# }}}

class Specificity(tf.keras.metrics.Metric):# {{{
  def __init__(self, thresholds=0.5, name='specificity', **kwargs):
    super(Specificity, self).__init__(name=name, **kwargs)
    self.specificity = self.add_weight(name='specificity', initializer='zeros')
    self.tn = self.add_weight(name='tn', initializer='zeros')
    self.fp = self.add_weight(name='fp', initializer='zeros')
    self.thresholds=thresholds

  def update_state(self, y_true, y_pred, sample_weight=None):
    min_delta=1e-6
    y_pred=tf.cast(tf.where(y_pred>self.thresholds,1,0),tf.int8)
    y_true=tf.cast(y_true,tf.int8)

    fp=tf.math.count_nonzero(y_pred*(1-y_true),dtype=tf.float32)
    tn=tf.math.count_nonzero((1-y_pred)*(1-y_true),dtype=tf.float32)

    self.fp.assign_add(fp)
    self.tn.assign_add(tn)

    self.specificity.assign(self.tn/(self.fp+self.tn+min_delta))

  def result(self):
    return self.specificity

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.specificity.assign(0.)
    self.fp.assign(0.)
    self.tn.assign(0.)
# }}}

class MCC(tf.keras.metrics.Metric):# {{{
  def __init__(self, thresholds=0.5, name='mcc', **kwargs):
    super(MCC, self).__init__(name=name, **kwargs)
    self.mcc = self.add_weight(name='mcc', initializer='zeros')
    self.tp = self.add_weight(name='tp', initializer='zeros')
    self.fp = self.add_weight(name='fp', initializer='zeros')
    self.fn = self.add_weight(name='fn', initializer='zeros')
    self.tn = self.add_weight(name='tn', initializer='zeros')
    self.thresholds=thresholds

  def update_state(self, y_true, y_pred, sample_weight=None):
    min_delta=1e-6
    y_pred=tf.cast(tf.where(y_pred>self.thresholds,1,0),tf.int8)
    y_true=tf.cast(y_true,tf.int8)

    tp=tf.math.count_nonzero(y_pred*y_true,dtype=tf.float32)
    fp=tf.math.count_nonzero(y_pred*(1-y_true),dtype=tf.float32)
    fn=tf.math.count_nonzero((1-y_pred)*y_true,dtype=tf.float32)
    tn=tf.math.count_nonzero((1-y_pred)*(1-y_true),dtype=tf.float32)

    self.tp.assign_add(tp)
    self.fp.assign_add(fp)
    self.fn.assign_add(fn)
    self.tn.assign_add(tn)

    self.mcc.assign((self.tp*self.tn-self.fp*self.fn)/(((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))**(1/2)+min_delta))

  def result(self):
    return self.mcc

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.mcc.assign(0.)
    self.tp.assign(0.)
    self.fp.assign(0.)
    self.fn.assign(0.)
    self.tn.assign(0.)
# }}}

