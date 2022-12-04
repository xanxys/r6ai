use std::fmt::{self, Display, Formatter};

pub const B_SIZE: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

fn opposite(c: Cell) -> Cell {
    match c {
        Cell::Empty => Cell::Empty,
        Cell::Black => Cell::White,
        Cell::White => Cell::Black,
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Board {
    pub side_black: bool,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Put(PutAction),
    Pass,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PutAction {
    pub x: usize,
    pub y: usize,
    pub c: Cell,
}

pub fn to_ix(x: usize, y: usize) -> usize {
    y * B_SIZE + x
}

impl Board {
    pub fn new() -> Board {
        let mut cs = vec![Cell::Empty; B_SIZE * B_SIZE];
        let ofs = (B_SIZE - 2) / 2;
        cs[to_ix(ofs + 0, ofs + 0)] = Cell::White;
        cs[to_ix(ofs + 1, ofs + 0)] = Cell::Black;
        cs[to_ix(ofs + 0, ofs + 1)] = Cell::Black;
        cs[to_ix(ofs + 1, ofs + 1)] = Cell::White;

        Board {
            side_black: true,
            cells: cs,
        }
    }

    pub fn is_terminal(&self) -> Option<Cell> {
        if self.possible_put_actions(Cell::Black).is_empty()
            && self.possible_put_actions(Cell::White).is_empty()
        {
            // ended
            let (n_b, n_w) = self.count_stones();
            if n_b > n_w {
                return Some(Cell::Black);
            } else if n_b < n_w {
                return Some(Cell::White);
            } else {
                return Some(Cell::Empty); // draw
            }
        } else {
            // ongoing
            return None;
        }
    }

    fn count_stones(&self) -> (i32, i32) {
        let mut black = 0;
        let mut white = 0;
        for c in &self.cells {
            match c {
                Cell::Empty => (),
                Cell::Black => black += 1,
                Cell::White => white += 1,
            }
        }
        (black, white)
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        firestorm::profile_fn!(legal_actions);
        let c = if self.side_black {
            Cell::Black
        } else {
            Cell::White
        };

        let puts = self.possible_put_actions(c);
        if puts.is_empty() {
            vec![Action::Pass]
        } else {
            puts.into_iter().map(|p| Action::Put(p)).collect()
        }
    }

    fn possible_put_actions(&self, c: Cell) -> Vec<PutAction> {
        let mut actions = vec![];
        for y in 0..B_SIZE {
            for x in 0..B_SIZE {
                let action = PutAction { x: x, y: y, c: c };
                if self.is_put_possible(&action) {
                    actions.push(action);
                }
            }
        }
        actions
    }

    pub fn is_legal(&self, action: &Action) -> bool {
        self.legal_actions().contains(action)
    }

    fn is_put_possible(&self, action: &PutAction) -> bool {
        if action.x >= B_SIZE || action.y >= B_SIZE {
            return false;
        }
        if self.cells[to_ix(action.x, action.y)] != Cell::Empty {
            return false;
        }
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if !self.can_take(action, dx, dy).is_empty() {
                    return true;
                }
            }
        }
        return false;
    }

    fn can_take(&self, action: &PutAction, dx: i32, dy: i32) -> Vec<usize> {
        let mut cx = action.x as i32 + dx;
        let mut cy = action.y as i32 + dy;

        enum Mode {
            Init,
            EnemyRun,
        }
        let mut may_take = vec![];
        let mut mode = Mode::Init;

        for i in 0..B_SIZE - 1 {
            if cx < 0 || cx >= B_SIZE as i32 || cy < 0 || cy >= B_SIZE as i32 {
                return vec![];
            }

            let curr_c = self.cells[to_ix(cx as usize, cy as usize)];
            match mode {
                Mode::Init => {
                    if curr_c == opposite(action.c) {
                        may_take.push(to_ix(cx as usize, cy as usize));
                        mode = Mode::EnemyRun;
                    } else {
                        return vec![];
                    }
                }
                Mode::EnemyRun => {
                    if curr_c == Cell::Empty {
                        return vec![];
                    } else if curr_c == opposite(action.c) {
                        // continue
                        may_take.push(to_ix(cx as usize, cy as usize));
                    } else {
                        // 1 or more enemy stones taken -> ok.
                        return may_take;
                    }
                }
            }
            cx += dx;
            cy += dy;
        }
        return vec![];
    }

    pub fn apply(&mut self, action: &Action) {
        firestorm::profile_fn!(apply);

        if !self.is_legal(action) {
            panic!("trying to apply illegal action");
        }

        if let Action::Put(put) = action {
            self.cells[to_ix(put.x, put.y)] = put.c;
            let mut takes: Vec<usize> = vec![];
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    takes.extend(self.can_take(put, dx, dy).iter());
                }
            }
            for ix in takes {
                self.cells[ix] = put.c;
            }
        }
        self.side_black = !self.side_black;
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        for y in 0..B_SIZE {
            for x in 0..B_SIZE {
                let c = match self.cells[to_ix(x, y)] {
                    Cell::Empty => ". ",
                    Cell::Black => "X ",
                    Cell::White => "O ",
                };
                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
