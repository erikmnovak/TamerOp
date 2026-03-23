# TamerOp

TamerOp is a Julia library for multiparameter persistence built from an encoding-first viewpoint. The central idea is that a multiparameter persistence module should be represented first by a finite, computable encoding on a finite poset, and then analyzed through that encoding. This perspective is guided by the theory of tame modules, especially the viewpoint developed by Ezra Miller: a multiparameter module is understood through finite combinatorial data that captures its essential structure while remaining mathematically faithful.

In practice, this means TamerOp treats encoding not as an implementation detail, but as the central mathematical bridge between raw data and downstream computation. Rather than committing early to one invariant or one storage format, the library emphasizes building finite encoded models that can support many later tasks. This makes it possible to organize multiparameter persistence workflows around a common discrete object instead of a collection of unrelated ad hoc pipelines.

TamerOp starts from several kinds of inputs. These include raw data such as point clouds, graphs, and images, as well as more algebraic inputs such as fringes, flanges, and other presentation-style objects. From those inputs, the library constructs finite-poset encodings that serve as the canonical computational model. Once that encoded model is available, TamerOp supports a broad range of outputs: invariant computations, signed measures, sliced and fibered constructions, homological algebra, derived-functor calculations, visualization, serialization, and related workflows.

This organization is meant to make the library useful both for computation and for mathematical experimentation. A user can begin from concrete data, move to an encoding, and then ask many different questions of the same encoded object. The same encoded perspective also supports more algebraic workflows, where the starting point is already a module presentation rather than a dataset. In both cases, the finite-poset encoding is the common language connecting input, computation, and output.

## Installation

