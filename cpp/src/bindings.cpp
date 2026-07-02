// =============================================================================
// bindings.cpp — pybind11 bridge exposing the C++ core to asyncio Python.
//
// GIL strategy:
//   * Single scalar deltas (apply_delta) are cheap enough that GIL churn
//     would dominate; they hold the GIL.
//   * apply_deltas_batch / on_trades_batch accept NumPy-compatible buffers
//     and release the GIL for the duration of the loop. The asyncio feed
//     handler drains a websocket frame into flat arrays and hands the whole
//     frame to C++ in one call — one interpreter round-trip per frame
//     instead of per level, and the event loop stays responsive while C++
//     chews through a burst.
//   * evaluate() is pure and allocation-free; it holds the GIL (its cost is
//     nanoseconds; releasing would cost more than it saves).
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "order_book.hpp"
#include "signal_generator.hpp"
#include "checkpoint.hpp"

namespace py = pybind11;
using namespace microcore;

PYBIND11_MODULE(microcore, m) {
    m.doc() = "C++ microstructure core: L2 order book, OBI/VWAP signals, "
              "mmap checkpointing";

    py::enum_<Side>(m, "Side")
        .value("BID", Side::BID)
        .value("ASK", Side::ASK);

    py::enum_<Regime>(m, "Regime")
        .value("NORMAL", Regime::NORMAL)
        .value("CEX_NIGHT_PASSIVE", Regime::CEX_NIGHT_PASSIVE)
        .value("DEX_NIGHT_EXIT_ONLY", Regime::DEX_NIGHT_EXIT_ONLY)
        .value("HALTED", Regime::HALTED);

    py::enum_<Action>(m, "Action")
        .value("NONE", Action::NONE)
        .value("ENTER_LONG", Action::ENTER_LONG)
        .value("EXIT_LONG", Action::EXIT_LONG)
        .value("PLACE_DEEP_BID", Action::PLACE_DEEP_BID)
        .value("CANCEL_ALL_BIDS", Action::CANCEL_ALL_BIDS)
        .value("TRAIL_STOP", Action::TRAIL_STOP);

    py::class_<Signal>(m, "Signal")
        .def_readonly("action", &Signal::action)
        .def_readonly("limit_price", &Signal::limit_price)
        .def_readonly("z", &Signal::z)
        .def_readonly("obi", &Signal::obi)
        .def_readonly("mid", &Signal::mid)
        .def_readonly("seq", &Signal::seq)
        .def("__repr__", [](const Signal& s) {
            return "Signal(action=" + std::to_string(int(s.action)) +
                   ", z=" + std::to_string(s.z) +
                   ", obi=" + std::to_string(s.obi) +
                   ", limit=" + std::to_string(s.limit_price) + ")";
        });

    py::class_<SignalConfig>(m, "SignalConfig")
        .def(py::init<>())
        .def_readwrite("entry_z", &SignalConfig::entry_z)
        .def_readwrite("exit_z", &SignalConfig::exit_z)
        .def_readwrite("obi_min", &SignalConfig::obi_min)
        .def_readwrite("obi_flip", &SignalConfig::obi_flip)
        .def_readwrite("night_k", &SignalConfig::night_k)
        .def_readwrite("min_sigma", &SignalConfig::min_sigma)
        .def_readwrite("min_volume", &SignalConfig::min_volume);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<double, size_t>(),
             py::arg("tick_size"), py::arg("obi_depth") = 20)
        .def("apply_delta", &OrderBook::apply_delta,
             py::arg("side"), py::arg("price"), py::arg("qty"))
        // Batched frame ingestion: sides/prices/qtys are parallel lists for
        // one websocket frame. GIL released while C++ applies the burst.
        .def("apply_deltas_batch",
             [](OrderBook& ob, const std::vector<int>& sides,
                const std::vector<double>& prices,
                const std::vector<double>& qtys) {
                 if (sides.size() != prices.size() || prices.size() != qtys.size())
                     throw py::value_error("batch arrays must be equal length");
                 py::gil_scoped_release release;
                 for (size_t i = 0; i < sides.size(); ++i)
                     ob.apply_delta(sides[i] == 0 ? Side::BID : Side::ASK,
                                    prices[i], qtys[i]);
             })
        .def("on_trade", &OrderBook::on_trade, py::arg("price"), py::arg("qty"))
        .def("on_trades_batch",
             [](OrderBook& ob, const std::vector<double>& prices,
                const std::vector<double>& qtys) {
                 if (prices.size() != qtys.size())
                     throw py::value_error("batch arrays must be equal length");
                 py::gil_scoped_release release;
                 for (size_t i = 0; i < prices.size(); ++i)
                     ob.on_trade(prices[i], qtys[i]);
             })
        .def("clear", &OrderBook::clear)
        .def_property_readonly("obi", &OrderBook::obi)
        .def_property_readonly("mid", &OrderBook::mid)
        .def_property_readonly("microprice", &OrderBook::microprice)
        .def_property_readonly("best_bid", &OrderBook::best_bid)
        .def_property_readonly("best_ask", &OrderBook::best_ask)
        .def_property_readonly("spread_bps", &OrderBook::spread_bps)
        .def_property_readonly("vwap", &OrderBook::vwap)
        .def_property_readonly("sigma", &OrderBook::sigma)
        .def_property_readonly("vwap_zscore", &OrderBook::vwap_zscore)
        .def("vwap_band", &OrderBook::vwap_band, py::arg("k"))
        .def_property_readonly("seq", &OrderBook::seq);

    py::class_<SignalGenerator>(m, "SignalGenerator")
        .def(py::init<SignalConfig>(), py::arg("config") = SignalConfig{})
        .def("set_regime", &SignalGenerator::set_regime)
        .def_property("position",
                      &SignalGenerator::position, &SignalGenerator::set_position)
        .def_property_readonly("regime", &SignalGenerator::regime)
        // `config` returns a reference to the generator's OWN config (not a
        // copy), so `sig.config.entry_z = 2.0` writes through to what
        // evaluate() reads. reference_internal ties the returned object's
        // lifetime to the parent generator.
        .def_property("config",
                      py::cpp_function(&SignalGenerator::config_mut,
                                       py::return_value_policy::reference_internal),
                      py::cpp_function(&SignalGenerator::set_config))
        .def("evaluate", &SignalGenerator::evaluate, py::arg("book"));

    py::class_<Checkpointer>(m, "Checkpointer")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def_property_readonly("ok", &Checkpointer::ok)
        .def("save", &Checkpointer::save, py::arg("book"), py::arg("signal"))
        .def("load", &Checkpointer::load, py::arg("book"), py::arg("signal"));
}
