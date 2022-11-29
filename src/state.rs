//! Data structs and traits to describe the state of a dynamical system

use fixed_map::{Key, Map};
use std::{
    collections::VecDeque,
    ops::{Index, IndexMut},
    rc::Rc,
};

use crate::variable::Variable;

/// Marker trait for Variable keys of [State] collections
pub trait VarKey: Copy + fixed_map::key::Key + std::fmt::Debug {}

/// Prognostic variables of the shallow water equations
#[derive(Copy, Clone, Key, Debug)]
pub enum SwmVars {
    U,
    V,
    ETA,
}

impl VarKey for SwmVars {}

/// State of a system.
///
/// The state is a collection of all prognostic variables of an set of differential equations, that are
/// the variables which for the solution of the system.
#[derive(Debug)]
pub struct State<K, V>(Map<K, V>)
where
    K: VarKey;

impl<K, V> AsRef<Map<K, V>> for State<K, V>
where
    K: VarKey,
{
    fn as_ref(&self) -> &Map<K, V> {
        &self.0
    }
}

impl<K, V> AsMut<Map<K, V>> for State<K, V>
where
    K: VarKey,
{
    fn as_mut(&mut self) -> &mut Map<K, V> {
        &mut self.0
    }
}

impl<K, V> Index<K> for State<K, V>
where
    K: VarKey,
{
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        &self
            .0
            .get(index)
            .expect(&format!("Variable {index:?} should be in state"))
    }
}

impl<K, V> IndexMut<K> for State<K, V>
where
    K: VarKey,
{
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.0
            .get_mut(index)
            .expect(&format!("Variable {index:?} should be in state"))
    }
}

/// Type alias for mapping variable id onto the grid on which the variable is defined.
pub type GridMap<K, G> = Map<K, Rc<G>>;

impl<K, V> State<K, V>
where
    K: VarKey,
{
    /// Create a new state from a grid mapping.
    pub fn new<const ND: usize>(grid_map: &GridMap<K, V::Grid>) -> Self
    where
        V: Variable<ND>,
    {
        State(grid_map.iter().map(|(k, g)| (k, V::zeros(g))).collect())
    }
}

/// Buffered factory object for new state objects
///
/// This object creates new state objects.
/// Additionally, it provides a buffer to store used state objects for later
/// reuse to reduce the number of memory allocations.
#[derive(Debug)]
pub struct StateFactory<const ND: usize, K, V>
where
    V: Variable<ND>,
    K: VarKey,
{
    grid_map: GridMap<K, V::Grid>,
    buffer: VecDeque<State<K, V>>,
}

impl<const ND: usize, K, V> StateFactory<ND, K, V>
where
    V: Variable<ND>,
    K: VarKey,
{
    /// Create a new StateFactory object
    pub fn new(grid_map: impl IntoIterator<Item = (K, Rc<V::Grid>)>) -> Self
    where
        V: Variable<ND>,
    {
        Self {
            grid_map: grid_map.into_iter().collect(),
            buffer: VecDeque::new(),
        }
    }

    /// Consume a StateFactory and return it with a modified capacity of the internal buffer.
    ///
    /// Note that the capacity is only approximately set and may slightly differ from the value of `capacity`.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.buffer.reserve(capacity);
        self
    }

    /// Create a new state object. This call involves memory allocation.
    fn make_state(&self) -> State<K, V> {
        State::new(&self.grid_map)
    }

    /// Return a state object. The object is either popped form the internal buffer of allocated.
    /// **No assumptions can be made about the content of the data contained in the state's variables**.
    pub fn get(&mut self) -> State<K, V> {
        self.buffer.pop_front().unwrap_or_else(|| self.make_state())
    }

    /// Push a state object to the internal buffer for later reuse.
    pub fn take(&mut self, state: State<K, V>) {
        self.buffer.push_back(state);
    }
}

/// An ordered collection of state objects, typically representing state at consecutive time steps.
///
/// `StateDeque` wraps a [VecDeque], however, unlike [VecDeque] a `StateDeque` has a fixed capacity and
/// will not grow beyond that. If no more objects can be added to the deque, the "oldest" element will be
/// dropped or consumed by a [StateFactory]'s internal buffer.
/// New elements will be added to the end of the deque.
#[derive(Debug)]
pub struct StateDeque<K, V>(VecDeque<State<K, V>>)
where
    K: VarKey;

impl<K, V> StateDeque<K, V>
where
    K: VarKey,
{
    /// Create a new StateDeque with a given fixed capacity.
    pub fn new(capacity: usize) -> StateDeque<K, V> {
        StateDeque(VecDeque::<State<K, V>>::with_capacity(capacity))
    }

    /// Add an element to the end of the deque. If the capacity is reached, the oldest element will be droped or
    /// handed over to the `consumer` for reuse.
    pub fn push<const ND: usize>(
        &mut self,
        elem: State<K, V>,
        consumer: Option<&mut StateFactory<ND, K, V>>,
    ) where
        V: Variable<ND>,
    {
        // buffer full
        if self.0.len() == self.0.capacity() {
            self.0.pop_front().and_then(|state| {
                if let Some(state_factory) = consumer {
                    state_factory.take(state);
                }
                Some(())
            });
        }
        self.0.push_back(elem);
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if there are no elements in the deque.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a [VecDeque<&V>] with references to a particular variable of all contained states.
    pub fn get_var(&self, var: K) -> VecDeque<&V> {
        VecDeque::from_iter(self.0.iter().map(|s| &s[var]))
    }
}

impl<K, V> AsRef<VecDeque<State<K, V>>> for StateDeque<K, V>
where
    K: VarKey,
{
    fn as_ref(&self) -> &VecDeque<State<K, V>> {
        &self.0
    }
}

impl<K, V> AsMut<VecDeque<State<K, V>>> for StateDeque<K, V>
where
    K: VarKey,
{
    fn as_mut(&mut self) -> &mut VecDeque<State<K, V>> {
        &mut self.0
    }
}

impl<K, V> Index<usize> for StateDeque<K, V>
where
    K: VarKey,
{
    type Output = State<K, V>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<K, V> IndexMut<usize> for StateDeque<K, V>
where
    K: VarKey,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

// #[cfg(test)]
// mod test {
//     use std::rc::Rc;

//     use fixed_map::Key;

//     use crate::{
//         field::{ArrND, Field},
//         grid::{Grid, GridND},
//         mask::DomainMask,
//         variable::Var,
//     };

//     use super::{StateDeque, StateFactory, VarKey};

//     #[derive(Clone, Copy, Key, Debug)]
//     enum SingleVar {
//         Var,
//     }

//     impl VarKey for SingleVar {}

//     fn make_simple_state_factory() -> StateFactory<
//         SingleVar,
//         Var<ArrND<2, f64>, GridND<2, f64, DomainMask>>,
//         GridND<2, f64, DomainMask>,
//     > {
//         let mut grid = GridND::cartesian([5, 5], 0f64, 0f64, 1.0, 1.0);
//         grid.with_mask(ArrND::full(DomainMask::Inside, [5, 5]));

//         let grid_map = (SingleVar::Var, Rc::new(grid));

//         StateFactory::new([grid_map])
//     }

//     #[test]
//     fn state_index_returns_variable() {
//         let state = make_simple_state_factory().make_state();

//         assert_eq!(state.0.len(), 1);

//         let _ = &state[SingleVar::Var];
//     }

//     #[test]
//     fn statedeque_index() {
//         let state_factory = make_simple_state_factory();

//         let mut state_deque = StateDeque::new(3);

//         state_deque.push(state_factory.make_state(), None);

//         let r = &state_deque[0];
//         println!("{:#?}", r);
//     }

//     #[test]
//     fn state_deque_return_states_to_consumer() {
//         let mut state_factory = make_simple_state_factory();
//         let mut state_deque = StateDeque::new(2);

//         let capacity = state_deque.as_ref().capacity();

//         for i in 0..2 * capacity {
//             state_deque.push(state_factory.make_state(), Some(&mut state_factory));
//             if i >= capacity {
//                 assert_eq!(state_factory.buffer.len(), i - capacity + 1);
//             } else {
//                 assert_eq!(state_factory.buffer.len(), 0);
//             }
//         }
//     }

//     #[test]
//     fn state_deque_cannot_grow_beyond_capacity() {
//         let state_factory = make_simple_state_factory();
//         let mut state_deque = StateDeque::new(2);

//         let capacity = state_deque.as_ref().capacity();

//         for i in 0..2 * capacity {
//             state_deque.push(state_factory.make_state(), None);
//             if i >= capacity {
//                 assert_eq!(state_deque.len(), capacity)
//             } else {
//                 assert_eq!(state_deque.len(), i + 1)
//             }
//         }
//     }
// }
