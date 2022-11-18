use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
    ops::{Index, IndexMut},
    rc::Rc,
};

use crate::{mask::Mask, variable::Variable};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SWMVars {
    U,
    V,
    ETA,
}

pub trait VarSet: Sized + Copy + Eq + Hash {
    fn values() -> &'static [Self];
}

impl VarSet for SWMVars {
    fn values() -> &'static [Self] {
        &[Self::U, Self::V, Self::ETA][..]
    }
}

#[derive(Debug)]
pub struct State<K, V> {
    vars: HashMap<K, V>,
}

#[derive(Debug, Clone)]
pub struct GridMap<K, G>(K, Rc<G>)
where
    K: Clone;

impl<K, V> State<K, V>
where
    K: VarSet,
{
    pub fn new<I, M>(grid_map: &[GridMap<K, <V as Variable<I, M>>::Grid>]) -> Self
    where
        V: Variable<I, M>,
        M: Mask,
    {
        State {
            vars: HashMap::from_iter(grid_map.iter().map(|GridMap(k, g)| (*k, V::zeros(g)))),
        }
    }
}

impl<K, V> Index<K> for State<K, V>
where
    K: Eq + Hash,
{
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        self.vars.get(&index).unwrap()
    }
}

impl<K, V> IndexMut<K> for State<K, V>
where
    K: Eq + Hash,
{
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.vars.get_mut(&index).unwrap()
    }
}

#[derive(Debug)]
pub struct StateFactory<K, V, I, M>
where
    V: Variable<I, M>,
    M: Mask,
    K: Clone,
{
    grid_map: Box<[GridMap<K, V::Grid>]>,
    buffer: VecDeque<State<K, V>>,
}

impl<K, V, I, M> StateFactory<K, V, I, M>
where
    V: Variable<I, M>,
    K: Copy,
    M: Mask,
{
    pub fn new(grid_map: impl Iterator<Item = GridMap<K, V::Grid>>) -> Self {
        Self {
            grid_map: grid_map.collect::<Vec<_>>().into_boxed_slice(),
            buffer: VecDeque::new(),
        }
    }

    pub fn from_state(state: State<K, V>) -> Self {
        let grid_map = state
            .vars
            .iter()
            .map(|(k, v)| GridMap(*k, v.get_grid().clone()));
        Self::new(grid_map)
    }

    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.buffer.reserve(capacity);
        self
    }

    pub fn make_state(&self) -> State<K, V>
    where
        K: VarSet,
    {
        State::new(&self.grid_map)
    }

    pub fn get(&mut self) -> State<K, V>
    where
        K: VarSet,
    {
        self.buffer.pop_front().unwrap_or_else(|| self.make_state())
    }

    pub fn take(&mut self, state: State<K, V>) {
        self.buffer.push_back(state);
    }
}

#[derive(Debug)]
pub struct StateDeque<K, V> {
    inner: VecDeque<State<K, V>>,
}

impl<K, V> StateDeque<K, V> {
    pub fn new(capacity: usize) -> StateDeque<K, V> {
        StateDeque {
            inner: VecDeque::<State<K, V>>::with_capacity(capacity),
        }
    }

    pub fn push<I, M>(&mut self, elem: State<K, V>, consumer: Option<&mut StateFactory<K, V, I, M>>)
    where
        V: Variable<I, M>,
        M: Mask,
        K: Copy,
    {
        // buffer full
        if self.inner.len() == self.inner.capacity() {
            self.inner.pop_front().and_then(|state| {
                if let Some(state_factory) = consumer {
                    state_factory.take(state);
                }
                Some(())
            });
        }
        self.inner.push_back(elem);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn get_var(&self, var: K) -> VecDeque<&V>
    where
        K: VarSet,
    {
        VecDeque::from_iter(self.inner.iter().map(|s| &s[var]))
    }
}

impl<K, V> AsRef<VecDeque<State<K, V>>> for StateDeque<K, V> {
    fn as_ref(&self) -> &VecDeque<State<K, V>> {
        &self.inner
    }
}

impl<K, V> Index<usize> for StateDeque<K, V> {
    type Output = State<K, V>;

    fn index(&self, index: usize) -> &Self::Output {
        self.inner.index(index)
    }
}

impl<K, V> IndexMut<usize> for StateDeque<K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.inner.index_mut(index)
    }
}

#[cfg(test)]
mod test {
    use std::rc::Rc;

    use crate::{
        field::{Arr2D, Field},
        grid::{Grid, Grid2D},
        mask::DomainMask,
        variable::Var,
    };

    use super::{GridMap, StateDeque, StateFactory, VarSet};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum SingleVar {
        Var,
    }

    impl VarSet for SingleVar {
        fn values() -> &'static [Self] {
            &[SingleVar::Var]
        }
    }

    fn make_simple_state_factory(
    ) -> StateFactory<SingleVar, Var<Arr2D<f64>, Grid2D<f64, DomainMask>>, f64, DomainMask> {
        let mut grid = Grid2D::cartesian((5, 5), 0f64, 0f64, 1.0, 1.0);
        grid.with_mask(Arr2D::full(DomainMask::Inside, (5, 5)));

        let grid_map: GridMap<_, _> = GridMap(SingleVar::Var, Rc::new(grid));

        StateFactory::new([grid_map].into_iter())
    }

    #[test]
    fn statedeque_index() {
        let state_factory = make_simple_state_factory();

        let mut state_deque = StateDeque::new(3);

        state_deque.push(state_factory.make_state(), None);

        let r = &state_deque[0];
        println!("{:#?}", r);
    }

    #[test]
    fn state_deque_return_states_to_consumer() {
        let mut state_factory = make_simple_state_factory();
        let mut state_deque = StateDeque::new(2);

        let capacity = state_deque.as_ref().capacity();

        for i in 0..2 * capacity {
            state_deque.push(state_factory.make_state(), Some(&mut state_factory));
            if i >= capacity {
                assert_eq!(state_factory.buffer.len(), i - capacity + 1);
            } else {
                assert_eq!(state_factory.buffer.len(), 0);
            }
        }
    }

    #[test]
    fn state_deque_cannot_grow_beyond_capacity() {
        let state_factory = make_simple_state_factory();
        let mut state_deque = StateDeque::new(2);

        let capacity = state_deque.as_ref().capacity();

        for i in 0..2 * capacity {
            state_deque.push(state_factory.make_state(), None);
            if i >= capacity {
                assert_eq!(state_deque.len(), capacity)
            } else {
                assert_eq!(state_deque.len(), i + 1)
            }
        }
    }
}
